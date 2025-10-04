"""
net_k8s_collectors.py

Kubernetes 상의 UERANSIM(RAN) + OAI Core 환경을 가정하고,
Prometheus(메트릭)와 Loki(로그)로부터 RAN/Transport/Core/App 지표를 수집하는 수집기.
일부 트랜스포트/CDN 측정은 외부 툴(iperf3, ping, traceroute, tc) 실행을 래핑.

필요:
  pip install requests
  (선택) pip install pandas

외부 툴(필요 시):
  - iperf3, ping, traceroute(or mtr), tc, tcptraceroute, owping(또는 twamp 관련 툴)

환경:
  - Prometheus/Loki에 접근 가능한 URL (포트포워딩 또는 Ingress)
  - Loki 라벨은 {namespace="oai"} 등 환경에 맞춰 조정

사용 예:
  python net_k8s_collectors.py --prom http://127.0.0.1:9090 --loki http://127.0.0.1:3100 \
    --ns oai --upf_pod oai-upf-0 --iface eth0 \
    --run all --window 15m --step 30s

"""
import argparse
import json
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

from secret import *
from Prome_helper import *

# ------------------ Shell helpers ------------------
def run(cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr

# ======================= RAN =======================
def parse_rsrp_sinr_from_logs(loki: LokiClient, ns: str, start_ns: int, end_ns: int,
                              ue_selector: str = '{app="ueransim-ue"}') -> List[Dict[str, Any]]:
    """
    UERANSIM UE 로그 또는 gNB 로그에서 MeasurementReport를 파싱하여 (RSRP, SINR) 추출
    LogQL 예: {namespace="oai", app="ueransim-ue"} |= "MeasurementReport"
    """
    query = f'{{namespace="{ns}"}} {ue_selector.replace("{", "|").replace("}", "|")} |= "MeasurementReport"'
    # 더 안전하게는 라벨 셀렉터를 합치는 것이 좋지만, 환경에 맞게 수정
    query = f'{{namespace="{ns}", app="ueransim-ue"}} |= "MeasurementReport"'
    resp = loki.query_range(query=query, start=start_ns, end=end_ns, limit=5000, direction="BACKWARD")
    results: List[Dict[str, Any]] = []
    # 예시 로그 패턴 (환경에 맞게 조정):
    # "RRC MeasurementReport: RSRP=-85 dBm, SINR=18 dB, PCI=123, CellId=..."
    rsrp_re = re.compile(r'RSRP[=\s:-]+(-?\d+)', re.IGNORECASE)
    sinr_re = re.compile(r'SINR[=\s:-]+(-?\d+)', re.IGNORECASE)
    ue_id_re = re.compile(r'UE[ _-]?ID[=\s:-]+(\S+)', re.IGNORECASE)

    for stream in resp.get("data", {}).get("result", []):
        labels = stream.get("stream", {})
        for ts, line in stream.get("values", []):
            rsrp = rsrp_re.search(line)
            sinr = sinr_re.search(line)
            ueid = ue_id_re.search(line)
            if rsrp or sinr:
                results.append({
                    "timestamp_ns": int(ts),
                    "ue_id": (ueid.group(1) if ueid else labels.get("pod")),
                    "rsrp_dbm": (int(rsrp.group(1)) if rsrp else None),
                    "sinr_db": (int(sinr.group(1)) if sinr else None),
                    **labels
                })
    return results


def rrc_state_counts(loki: LokiClient, ns: str, start_ns: int, end_ns: int,
                     gnb_selector: str = '{app="oai-gnb"}') -> Dict[str, int]:
    """
    AMF/gNB 로그에서 RRC 상태 변화를 카운트 (CONNECTED/IDLE 등)
    예시 키워드: "RRC_CONNECTED", "RRC_IDLE"
    """
    query = f'{{namespace="{ns}", app="oai-gnb"}} |= "RRC_"'
    resp = loki.query_range(query=query, start=start_ns, end=end_ns, limit=5000, direction="BACKWARD")
    counts = {"CONNECTED": 0, "IDLE": 0, "INACTIVE": 0}
    for stream in resp.get("data", {}).get("result", []):
        for ts, line in stream.get("values", []):
            if "RRC_CONNECTED" in line:
                counts["CONNECTED"] += 1
            elif "RRC_IDLE" in line:
                counts["IDLE"] += 1
            elif "RRC_INACTIVE" in line or "INACTIVE" in line:
                counts["INACTIVE"] += 1
    return counts


def cell_congestion_level(prom: PrometheusClient, prb_metric: str, ns: str, gnb: Optional[str],
                          start_rfc3339: str, end_rfc3339: str, step: str = "30s") -> Dict[str, Any]:
    """
    gNB PRB 사용률 기반 혼잡도 계산. Prometheus에 노출되는 지표명이 환경마다 다름.
    예: prb_metric="oai_gnb_prb_used_ratio" (0~1)
    """
    match = f'{prb_metric}{{namespace="{ns}"' + (f', pod="{gnb}"' if gnb else "") + "}}"
    resp = prom.query_range(match, start=start_rfc3339, end=end_rfc3339, step=step)
    # 간단 혼잡도 규칙: <=0.5: Low, <=0.8: Medium, >0.8: High (시간 평균)
    vals = []
    for it in resp.get("data", {}).get("result", []):
        for ts, v in it.get("values", []):
            try:
                vals.append(float(v))
            except Exception:
                pass
    avg = sum(vals)/len(vals) if vals else 0.0
    if avg > 0.8:
        level = "High"
    elif avg > 0.5:
        level = "Medium"
    else:
        level = "Low"
    return {"avg_prb_usage": avg, "level": level, "samples": len(vals)}
'''
# =================== TRANSPORT =====================
def one_way_latency_udp(sender: Optional[str], receiver: Optional[str], count: int = 100, port: int = 8622,
                        interval_ms: int = 20, timeout: int = 5) -> Dict[str, Any]:
    """
    매우 단순한 UDP 원웨이 레이턴시 측정 래퍼 (NTP 동기화 전제).
    - sender/receiver 중 하나를 None으로 주고 반대편에서 별도 실행 필요.
    여기서는 외부 스크립트를 가정하지 않고, 동일 호스트에서 측정할 수 없으므로 Placeholder.
    실제 운영에선 twamp/owping 권장. 본 함수는 지표 스키마만 맞춰 반환.
    """
    return {"note": "Use TWAMP/owping in production. This is a placeholder.", "latency_ms_p50": None, "latency_ms_p95": None}


def iperf3_udp_loss(server: str, duration: int = 10, bandwidth: str = "50M", port: int = 5201) -> Dict[str, Any]:
    """
    로컬에서 iperf3 클라이언트를 실행하여 서버와 UDP 테스트. 서버는 미리 실행 필요:
      iperf3 -s -p 5201
    예:
      iperf3 -u -c <server> -b 50M -t 10 -p 5201 -J
    """
    cmd = ["iperf3", "-u", "-c", server, "-b", bandwidth, "-t", str(duration), "-p", str(port), "-J"]
    code, out, err = run(cmd, timeout=duration+10)
    if code != 0:
        return {"error": err or out}
    try:
        data = json.loads(out)
        end = data.get("end", {})
        sum_b = end.get("sum", {}) or end.get("sum_sent", {})
        loss = sum_b.get("lost_percent")
        jitter_ms = sum_b.get("jitter_ms")
        bps = sum_b.get("bits_per_second")
        return {"loss_percent": loss, "jitter_ms": jitter_ms, "throughput_bps": bps}
    except Exception as e:
        return {"error": f"parse failed: {e}"}


def jitter_from_timestamps(timestamps_ms: List[float]) -> Optional[float]:
    if not timestamps_ms:
        return None
    diffs = [t2 - t1 for t1, t2 in zip(timestamps_ms[:-1], timestamps_ms[1:])]
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    return var ** 0.5


def queue_occupancy_tc(iface: str) -> Dict[str, Any]:
    """
    tc -s qdisc show dev <iface> 출력 파싱하여 backlog/qlen 추정
    """
    code, out, err = run(["tc", "-s", "qdisc", "show", "dev", iface], timeout=5)
    if code != 0:
        return {"error": err or out}
    backlog_re = re.compile(r"backlog\s+(\d+)p\s+(\d+)b")
    qlen_re = re.compile(r"qlen\s+(\d+)")
    m_backlog = backlog_re.search(out)
    m_qlen = qlen_re.search(out)
    return {
        "backlog_pkts": int(m_backlog.group(1)) if m_backlog else None,
        "backlog_bytes": int(m_backlog.group(2)) if m_backlog else None,
        "qlen_pkts": int(m_qlen.group(1)) if m_qlen else None,
        "raw": out,
    }
'''
# ================== CORE NETWORK ===================
def pdu_session_delay(loki: LokiClient, ns: str, imsi: Optional[str], start_ns: int, end_ns: int) -> Dict[str, Any]:
    """
    AMF/SMF/UPF 각 로그에서 동일 세션(IMSI/SUPI/SessionID)의 주요 이벤트 타임스탬프 간 차이를 계산.
    패턴은 환경 로그 포맷에 맞게 조정 필요.
    """
    patterns = {
        "amf_req": re.compile(r"(PDU Session Establishment Request|NAS\spdu.*establish)", re.IGNORECASE),
        "smf_sel": re.compile(r"Select SMF|Nsmf_PDUSession", re.IGNORECASE),
        "upf_rule": re.compile(r"Create PDR|Create FAR|N4 Session", re.IGNORECASE),
        "amf_accept": re.compile(r"(PDU Session Establishment Accept|NAS\spdu.*accept)", re.IGNORECASE),
    }
    label_sel = f'namespace="{ns}"'
    q = f'{{{label_sel}}} |= "PDU Session" or |= "N4" or |= "Nsmf"'
    if imsi:
        q = f'{{{label_sel}}} |= "{imsi}"'
    resp = loki.query_range(query=q, start=start_ns, end=end_ns, limit=10000, direction="BACKWARD")
    t_amf_req = t_smf = t_upf = t_accept = None
    for stream in resp.get("data", {}).get("result", []):
        for ts, line in stream.get("values", []):
            ts_i = int(ts)
            if t_amf_req is None and patterns["amf_req"].search(line):
                t_amf_req = ts_i
            elif t_smf is None and patterns["smf_sel"].search(line):
                t_smf = ts_i
            elif t_upf is None and patterns["upf_rule"].search(line):
                t_upf = ts_i
            elif t_accept is None and patterns["amf_accept"].search(line):
                t_accept = ts_i
    def ms(ns_i): return (ns_i/1e6) if ns_i else None
    return {
        "amf_req_to_smf_ms": ms(t_smf) - ms(t_amf_req) if t_smf and t_amf_req else None,
        "smf_to_upf_ms": ms(t_upf) - ms(t_smf) if t_upf and t_smf else None,
        "upf_to_accept_ms": ms(t_accept) - ms(t_upf) if t_accept and t_upf else None,
        "end_to_end_ms": ms(t_accept) - ms(t_amf_req) if t_accept and t_amf_req else None,
        "timestamps": {"amf_req": t_amf_req, "smf": t_smf, "upf": t_upf, "accept": t_accept},
    }


def amf_registration_rate(loki: LokiClient, ns: str, start_ns: int, end_ns: int) -> Dict[str, Any]:
    """
    AMF 로그에서 Registration 성공/실패 카운트.
    예시 키워드: "Registration accept", "Registration reject"
    """
    resp = loki.query_range(query=f'{{namespace="{ns}", app="oai-amf"}} |= "Registration"', start=start_ns, end=end_ns, limit=10000, direction="BACKWARD")
    ok = fail = 0
    for stream in resp.get("data", {}).get("result", []):
        for ts, line in stream.get("values", []):
            low = line.lower()
            if "accept" in low:
                ok += 1
            elif "reject" in low or "failure" in low:
                fail += 1
    total = ok + fail
    rate = (ok / total) if total else None
    return {"ok": ok, "fail": fail, "rate": rate}


def upf_userplane_throughput(prom: PrometheusClient, ns: str, upf_pod: Optional[str],
                             metric_tx: str = 'container_network_transmit_bytes_total',
                             metric_rx: str = 'container_network_receive_bytes_total',
                             step: str = "30s", window: str = "5m") -> Dict[str, Any]:
    """
    UPF Pod 네트워크 인터페이스 기준의 전송량을 PromQL로 계산.
    환경에 맞춰 VPP/gtp5g exporter가 있으면 그 지표명을 사용.
    """
    sel = f'{{namespace="{ns}"' + (f', pod="{upf_pod}"' if upf_pod else "") + "}}"
    promql_tx = f'rate({metric_tx}{sel}[{window}])'
    promql_rx = f'rate({metric_rx}{sel}[{window}])'
    end = to_rfc3339()
    start = to_rfc3339(datetime.now(timezone.utc) - timedelta(minutes=15))
    r_tx = prom.query_range(promql_tx, start=start, end=end, step=step)
    r_rx = prom.query_range(promql_rx, start=start, end=end, step=step)
    def last_avg(resp):
        vals = []
        for it in resp.get("data", {}).get("result", []):
            series_vals = [float(v) for ts, v in it.get("values", []) if v not in ("NaN", "Inf")]
            if series_vals:
                vals.append(sum(series_vals)/len(series_vals))
        return sum(vals)/len(vals) if vals else 0.0
    return {"tx_bps": last_avg(r_tx), "rx_bps": last_avg(r_rx)}


def smf_session_drop_count(loki: LokiClient, ns: str, start_ns: int, end_ns: int) -> Dict[str, Any]:
    """
    SMF 제어 로그에서 세션 drop/release 판단. 키워드 조정 필요.
    """
    resp = loki.query_range(query=f'{{namespace="{ns}", app="oai-smf"}} |= "Session" ', start=start_ns, end=end_ns, limit=10000, direction="BACKWARD")
    drops = 0
    for stream in resp.get("data", {}).get("result", []):
        for ts, line in stream.get("values", []):
            if "drop" in line.lower() or "release" in line.lower():
                drops += 1
    return {"smf_session_drop": drops}

# ============ APPLICATION / SERVICE =================
# Need to implement measurement method in application#
# Really possible?
def qoe_score_simple(http_rtt_ms: float, rebuffer_count: int, resolution_changes: int) -> float:
    """
    간단 QoE 스코어 (5 점 만점) 예시 모델:
      base = 5.0 - 0.001*RTT - 0.5*rebuffer - 0.2*res_changes
    """
    score = 5.0 - 0.001*http_rtt_ms - 0.5*rebuffer_count - 0.2*resolution_changes
    return max(1.0, min(5.0, score))


def qoe_from_logs(loki: LokiClient, ns: str, start_ns: int, end_ns: int) -> Dict[str, Any]:
    """
    애플리케이션 로그에서 RTT/rebuffer/resolution 변경 이벤트를 추출하여 QoE 산출.
    로그 포맷에 맞게 키워드 조정.
    """
    resp = loki.query_range(query=f'{{namespace="{ns}", app="http-client"|app="video-player"}}', start=start_ns, end=end_ns, limit=10000, direction="BACKWARD")
    rtts = []
    rebuf = 0
    reschg = 0
    rtt_re = re.compile(r"RTT[=\s:]+(\d+(\.\d+)?)\s*ms", re.IGNORECASE)
    for stream in resp.get("data", {}).get("result", []):
        for ts, line in stream.get("values", []):
            m = rtt_re.search(line)
            if m:
                rtts.append(float(m.group(1)))
            if "rebuffer" in line.lower():
                rebuf += 1
            if "resolution change" in line.lower() or "quality change" in line.lower():
                reschg += 1
    http_rtt_ms = sum(rtts)/len(rtts) if rtts else 0.0
    score = qoe_score_simple(http_rtt_ms, rebuf, reschg)
    return {"http_rtt_ms_avg": http_rtt_ms, "rebuffer_events": rebuf, "resolution_changes": reschg, "qoe_score": score}


def application_delay_from_har(har_path: str) -> Dict[str, Any]:
    """
    HAR 파일에서 요청-응답 RTT 통계 계산.
    """
    try:
        with open(har_path, "r", encoding="utf-8") as f:
            har = json.load(f)
    except Exception as e:
        return {"error": f"read har failed: {e}"}
    timings = []
    for entry in har.get("log", {}).get("entries", []):
        tms = entry.get("timings", {})
        total = tms.get("wait", 0) + tms.get("receive", 0) + tms.get("send", 0)
        if total >= 0:
            timings.append(total)
    if not timings:
        return {"count": 0}
    avg = sum(timings)/len(timings)
    p95 = sorted(timings)[int(0.95*len(timings))-1]
    return {"count": len(timings), "rtt_ms_avg": avg, "rtt_ms_p95": p95}


def buffer_underrun_from_logs(loki: LokiClient, ns: str, start_ns: int, end_ns: int) -> Dict[str, Any]:
    resp = loki.query_range(query=f'{{namespace="{ns}", app="video-player"}} |= "buffer underrun" or |= "Buffering"', start=start_ns, end=end_ns, limit=10000, direction="BACKWARD")
    cnt = 0
    for stream in resp.get("data", {}).get("result", []):
        cnt += len(stream.get("values", []))
    return {"buffer_underrun_events": cnt}


def cdn_rtt_targets(hosts: List[str], count: int = 5) -> List[Dict[str, Any]]:
    results = []
    for h in hosts:
        code, out, err = run(["ping", "-c", str(count), h], timeout=10+count)
        if code != 0:
            results.append({"host": h, "error": err or out})
            continue
        # rtt min/avg/max/mdev = 9.023/10.245/12.331/0.421 ms
        m = re.search(r"rtt .* = ([\d.]+)/([\d.]+)/([\d.]+)/", out)
        if m:
            results.append({"host": h, "rtt_ms_avg": float(m.group(2))})
        else:
            results.append({"host": h, "raw": out})
    return results

# ======================= CLI =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prom", default= prometheus_ip, help="Prometheus base URL")
    ap.add_argument("--loki", default= loki_ip, help="Loki base URL")
    ap.add_argument("--ns", default='oai', help="Kubernetes namespace (e.g., oai)")
    ap.add_argument("--upf_pod", help="UPF pod name label value (optional)")
    ap.add_argument("--iface", default="eth0", help="Interface for tc stats")
    ap.add_argument("--window", default="15m", help="Lookback window (e.g., 15m)")
    ap.add_argument("--step", default="30s", help="Prometheus step")
    ap.add_argument("--run", default="all", help="What to run: ran,transport,core,app,all")
    args = ap.parse_args()

    prom = PrometheusClient(args.prom)
    loki = LokiClient(args.loki)

    end_rfc = to_rfc3339()
    start_rfc = to_rfc3339(datetime.now(timezone.utc) - timedelta(minutes=int(args.window.rstrip("m"))))
    end_ns = now_ns()
    start_ns = now_ns(minutes(-int(args.window.rstrip("m"))))

    out: Dict[str, Any] = {}

    if args.run in ("ran", "all"):
        out["ran"] = {
            "rsrp_sinr": parse_rsrp_sinr_from_logs(loki, ns=args.ns, start_ns=start_ns, end_ns=end_ns),
            "rrc_state_counts": rrc_state_counts(loki, ns=args.ns, start_ns=start_ns, end_ns=end_ns),
            "cell_congestion": cell_congestion_level(prom, prb_metric="oai_gnb_prb_used_ratio", ns=args.ns, gnb=None,
                                                     start_rfc3339=start_rfc, end_rfc3339=end_rfc, step=args.step),
        }
    '''
    if args.run in ("transport", "all"):
        out["transport"] = {
            "udp_iperf3": {"note": "Run iperf3 server before using", **iperf3_udp_loss("127.0.0.1")},
            "queue": queue_occupancy_tc(args.iface),
        }
    '''

    if args.run in ("core", "all"):
        out["core"] = {
            "pdu_session_delay": pdu_session_delay(loki, ns=args.ns, imsi=None, start_ns=start_ns, end_ns=end_ns),
            "amf_registration_rate": amf_registration_rate(loki, ns=args.ns, start_ns=start_ns, end_ns=end_ns),
            "upf_throughput": upf_userplane_throughput(prom, ns=args.ns, upf_pod=args.upf_pod),
            "smf_session_drop": smf_session_drop_count(loki, ns=args.ns, start_ns=start_ns, end_ns=end_ns),
        }

    if args.run in ("app", "all"):
        out["app"] = {
            "qoe": qoe_from_logs(loki, ns='application', start_ns=start_ns, end_ns=end_ns),
            "buffer_underrun": buffer_underrun_from_logs(loki, ns=args.ns, start_ns=start_ns, end_ns=end_ns),
            "cdn_rtt": cdn_rtt_targets(["edge.example.cdn", "edge2.example.cdn"]),
        }

    print(p.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
