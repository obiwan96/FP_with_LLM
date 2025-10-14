#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obs_client.py - Prometheus / Loki HTTP API wrapper (Python 3.9 compatible)
"""
import dataclasses
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union  # 3.9 호환

import requests
from requests.auth import HTTPBasicAuth
import re

# ------------------------------
# 시간 헬퍼
# ------------------------------
def to_rfc3339(dt: Optional[datetime] = None) -> str:
    """datetime -> RFC3339 문자열(UTC). dt가 None이면 현재 시간."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

def to_epoch(dt: Optional[datetime] = None) -> float:
    """datetime -> epoch seconds (float). dt가 None이면 현재."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()

def now_rfc3339(offset: Optional[timedelta] = None) -> str:
    base = datetime.now(timezone.utc)
    if offset:
        base += offset
    return base.isoformat()

def now_epoch(offset: Optional[timedelta] = None) -> float:
    base = datetime.now(timezone.utc)
    if offset:
        base += offset
    return base.timestamp()

def now_ns(offset: Optional[timedelta] = None) -> int:
    """현재 시간(또는 오프셋 적용) ns epoch 정수값"""
    base = datetime.now(timezone.utc)
    if offset:
        base += offset
    return int(base.timestamp() * 1_000_000_000)

def minutes(n: int) -> timedelta:
    return timedelta(minutes=n)

def hours(n: int) -> timedelta:
    return timedelta(hours=n)

def days(n: int) -> timedelta:
    return timedelta(days=n)

def ns(seconds: float) -> int:
    """초 단위를 ns 정수로"""
    return int(seconds * 1_000_000_000)

# ------------------------------
# HTTP 베이스 클라이언트
# ------------------------------
@dataclasses.dataclass
class HttpConfig:
    base_url: str
    username: Optional[str] = None
    password: Optional[str] = None
    bearer_token: Optional[str] = None
    headers: Optional[dict] = None
    verify_tls: bool = True
    timeout: int = 20
    retries: int = 2
    backoff: float = 0.5  # seconds

class BaseHttpClient:
    def __init__(self, http: HttpConfig):
        self.http = http

    def _auth(self):
        if self.http.username and self.http.password:
            return HTTPBasicAuth(self.http.username, self.http.password)
        return None

    def _headers(self) -> dict:
        h = {"Accept": "application/json"}
        if self.http.headers:
            h.update(self.http.headers)
        if self.http.bearer_token:
            h["Authorization"] = f"Bearer {self.http.bearer_token}"
        return h

    def _request(self, method: str, path: str, params: Optional[dict] = None) -> dict:
        url = self.http.base_url.rstrip("/") + "/" + path.lstrip("/")
        for attempt in range(1, self.http.retries + 2):
            try:
                r = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=self._headers(),
                    auth=self._auth(),
                    timeout=self.http.timeout,
                    verify=self.http.verify_tls,
                )
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt <= self.http.retries:
                    time.sleep(self.http.backoff * attempt)
                else:
                    raise

# ------------------------------
# Core function log reading
# ------------------------------

def extract_supi(line: str) -> str:
    """
    Extract SUPI (IMSI) from a log line.
    Examples:
      - "SUPI imsi-001010000000106" → "001010000000106"
      - "SUPI 1010000000106" → "1010000000106"
    """
    match = re.search(r"SUPI[\s:=\-]*((imsi-)?(\d{10,15}))", line)
    if match:
        return int(match.group(3))  # 숫자만 반환
    return None

def get_amf_request_times(amf_logs: List[tuple[datetime, str]]) -> Dict[str, datetime]:
    """
    Extract T_request per SUPI from AMF logs.
    """
    supi_to_time = {}
    for ts, line in amf_logs:
        if "Handle PDU Session Establishment Request" in line:
            supi = extract_supi(line)
            #print(f'find supi {supi} in amf log')
            if supi and supi not in supi_to_time:
                supi_to_time[supi] = ts
    return supi_to_time

def get_smf_accept_times(smf_logs: List[tuple[datetime, str]]) -> Dict[str, datetime]:
    """
    Extract T_accept per SUPI from SMF logs.
    """
    supi_to_time = {}
    for ts, line in smf_logs:
        if "triger PDU_SES_EST" in line:
            supi = extract_supi(line)
            #print(f'find supi {supi} in smf log')
            if supi and supi not in supi_to_time:
                supi_to_time[supi] = ts
    return supi_to_time


# ------------------------------
# Prometheus
# ------------------------------

''' Usage:
prom = PrometheusClient("http://localhost:9090")

start = to_rfc3339(datetime.now(timezone.utc) + minutes(-10))  # 10분 전
end = to_rfc3339()                # 현재

metric_names = prom.label_list(start, end)
print("활성 metric 목록:")
for name in metric_names:
    print(name)
'''
class PrometheusClient(BaseHttpClient):
    def __init__(self, base_url: str, **kwargs):
        super().__init__(HttpConfig(base_url=base_url, **kwargs))

    def label_list(self, start: Union[float, str], end: Union[float, str]) -> List[str]:
        try:
            resp = self.series(match=["{__name__=~'.+'}"], start=start, end=end)
            data = resp.get("data", [])
            return sorted({item["__name__"] for item in data if "__name__" in item})
        except Exception as e:
            print(f"[active_metrics] Error: {e}")
            return []


    def query(self, promql: str,
              time_: Optional[Union[float, str]] = None,
              timeout: Optional[str] = None) -> dict:
        params = {"query": promql}
        if time_ is not None:
            params["time"] = time_ if isinstance(time_, str) else f"{time_:.3f}"
        if timeout:
            params["timeout"] = timeout
        return self._request("GET", "/api/v1/query", params)

    def query_range(self,
                    promql: str,
                    start: Union[float, str],
                    end: Union[float, str],
                    step: Union[str, float],
                    timeout: Optional[str] = None) -> dict:
        params = {"query": promql, "start": start, "end": end, "step": step}
        if timeout:
            params["timeout"] = timeout
        return self._request("GET", "/api/v1/query_range", params)

    def series(self, match: List[str],
               start: Optional[Union[float, str]] = None,
               end: Optional[Union[float, str]] = None) -> dict:
        params: List[tuple] = [("match[]", m) for m in match]
        url = self.http.base_url.rstrip("/") + "/api/v1/series"
        if start is not None:
            params.append(("start", str(start)))
        if end is not None:
            params.append(("end", str(end)))

        r = requests.get(
            url,
            params=params,
            headers=self._headers(),
            auth=self._auth(),
            timeout=self.http.timeout,
            verify=self.http.verify_tls,
        )
        r.raise_for_status()
        return r.json()

# ------------------------------
# Loki
# usage:
# client = LokiClient("http://localhost:3100")
# containers = client.get_containers_recently_logged("oai")
# print(containers)  # 예: ['upf', 'amf', 'smf']
# ------------------------------

class LokiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _to_nanots(self, dt: datetime) -> int:
        return int(dt.timestamp() * 1e9)

    def query_range(self, query: str, start: int, end: int, limit: int = 1000, direction: str = "backward"):
        params = {
            "query": query,
            "start": start,
            "end": end,
            "limit": limit,
            "direction": direction,
        }
        response = requests.get(f"{self.base_url}/loki/api/v1/query_range", params=params)
        response.raise_for_status()
        return response.json()

    def get_containers_recently_logged(self, namespace: str):
        now = datetime.utcnow()
        start = now - timedelta(minutes=5)
        logql = f'{{namespace="{namespace}"}}'
        result = self.query_range(logql, start, now)

        container_set = set()
        for stream in result.get("data", {}).get("result", []):
            labels = stream.get("stream", {})
            container = labels.get("container")
            if container:
                container_set.add(container)
        return list(container_set)

    def get_recent_logs(self, namespace: str, container: str=None, limit: int = 2000, window: int=5):
        now = now_ns()
        start = now_ns(minutes(-window))
        if container:
            logql = f'{{namespace="{namespace}", container="{container}"}}'
        else:
            logql = f'{{namespace="{namespace}"}}'
        result = self.query_range(logql, start, now, limit=limit)

        logs = []
        for stream in result.get("data", {}).get("result", []):
            container = stream.get("stream", {}).get("container", "unknown")
            for ts, log in stream.get("values", []):
                log_time = datetime.fromtimestamp(int(ts) / 1e9)
                logs.append({
                    "timestamp": log_time,
                    "container": container,
                    "log": log.strip()
                })
        return logs


# ------------------------------
# (선택) pandas 변환
# ------------------------------
def maybe_pandas_dataframe_from_prometheus(resp: dict):
    try:
        import pandas as pd
    except Exception:
        return None

    if resp.get("status") != "success":
        return None
    data = resp.get("data", {})
    result_type = data.get("resultType")
    result = data.get("result", [])

    if result_type == "vector":
        rows = []
        for item in result:
            metric = item.get("metric", {})
            v = item.get("value", [])
            ts = float(v[0]) if v else None
            val = float(v[1]) if v else None
            rows.append({"timestamp": datetime.utcfromtimestamp(ts), "value": val, **metric})
        return pd.DataFrame(rows)

    if result_type == "matrix":
        rows = []
        for item in result:
            metric = item.get("metric", {})
            for ts, val in item.get("values", []):
                rows.append({"timestamp": datetime.utcfromtimestamp(float(ts)), "value": float(val), **metric})
        return pd.DataFrame(rows)

    return None

def maybe_pandas_dataframe_from_loki(resp: dict):
    try:
        import pandas as pd
    except Exception:
        return None

    data = resp.get("data", {})
    streams = data.get("result", [])
    rows = []
    for stream in streams:
        labels = stream.get("stream", {})
        for ts_ns, line in stream.get("values", []):
            ts = datetime.fromtimestamp(int(ts_ns) / 1_000_000_000, tz=timezone.utc)
            rows.append({"timestamp": ts, "line": line, **labels})
    if not rows:
        return pd.DataFrame(columns=["timestamp", "line"])
    return pd.DataFrame(rows)