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

__all__ = [
    "PrometheusClient",
    "LokiClient",
    "to_rfc3339",
    "to_epoch",
    "now_rfc3339",
    "now_epoch",
    "now_ns",
    "minutes",
    "hours",
    "days",
    "ns",
    "maybe_pandas_dataframe_from_prometheus",
    "maybe_pandas_dataframe_from_loki",
]

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
# Prometheus
# ------------------------------

''' Usage:ㄴ
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
# ------------------------------
class LokiClient(BaseHttpClient):
    def __init__(self, base_url: str, **kwargs):
        super().__init__(HttpConfig(base_url=base_url, **kwargs))

    def query(self, query: str,
              time_: Optional[int] = None,
              limit: Optional[int] = None,
              direction: Optional[str] = None) -> dict:
        params: Dict[str, Any] = {"query": query}
        if time_ is not None:
            params["time"] = str(time_)
        if limit is not None:
            params["limit"] = int(limit)
        if direction:
            params["direction"] = direction
        return self._request("GET", "/loki/api/v1/query", params)

    def query_range(self,
                    query: str,
                    start: Optional[int] = None,
                    end: Optional[int] = None,
                    step: Optional[str] = None,
                    limit: Optional[int] = None,
                    direction: Optional[str] = None,
                    regexp: Optional[str] = None) -> dict:
        params: Dict[str, Any] = {"query": query}
        if start is not None:
            params["start"] = str(start)
        if end is not None:
            params["end"] = str(end)
        if step:
            params["step"] = step
        if limit is not None:
            params["limit"] = int(limit)
        if direction:
            params["direction"] = direction
        if regexp:
            params["regexp"] = regexp
        return self._request("GET", "/loki/api/v1/query_range", params)

    def labels(self, start: Optional[int] = None, end: Optional[int] = None) -> dict:
        params: Dict[str, Any] = {}
        if start is not None:
            params["start"] = str(start)
        if end is not None:
            params["end"] = str(end)
        return self._request("GET", "/loki/api/v1/labels", params)

    def series(self, match: List[str],
               start: Optional[int] = None,
               end: Optional[int] = None) -> dict:
        params: List[tuple] = [("match[]", m) for m in match]
        if start is not None:
            params.append(("start", str(start)))
        if end is not None:
            params.append(("end", str(end)))

        url = self.http.base_url.rstrip("/") + "/loki/api/v1/series"
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