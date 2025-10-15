import time
import requests
import datetime
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import os

TARGET_URL = "http://10.98.186.111:80"
CHECK_INTERVAL = 10  # seconds
SLO_LATENCY = 0.2  # seconds
SLO_FAIL_RATIO = 0.01

INFLUX_URL   = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN") 
alarm_url = os.getenv("ALARM_URL")
alarm_chat_id = os.getenv("ALARM_CHAT_ID")
alarm_body = {"chat_id":alarm_chat_id, "text": 'SLO Violation occured!' }
alarm_headers= { 'Content-Type': 'application/json' }
ORG = "dpnm"
BUCKET = "mdaf"

client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

def check_web_health():
    total, fail = 0, 0
    latencies = []
    for _ in range(5):
        start = time.time()
        try:
            r = requests.get(TARGET_URL, timeout=3)
            latency = time.time() - start
            latencies.append(latency)
            total += 1
            if r.status_code != 200:
                fail += 1
        except Exception:
            fail += 1
    return total, fail, latencies

def write_slo_violation():
    point = (
        Point("failure_history")
        .field("failure_history", "slo violation")
        .time(datetime.datetime.utcnow(), WritePrecision.NS)
    )
    write_api.write(bucket=BUCKET, record=point)
    print("[!] SLO violation recorded to InfluxDB")

def main():
    point = (
        Point("failure_history")
        .field("test", "fault_detector UE deployed well!")
        .time(datetime.datetime.utcnow(), WritePrecision.NS)
    )
    write_api.write(bucket=BUCKET, record=point)
    while True:
        total, fail, latencies = check_web_health()
        if total == 0:
            print("[WARN] NGINX unreachable")
            write_slo_violation()
        else:
            avg_lat = sum(latencies) / total
            fail_ratio = fail / total
            if avg_lat > SLO_LATENCY or fail_ratio > SLO_FAIL_RATIO:
                print(f"[ALERT] SLO violation: latency={avg_lat:.3f}, fail_ratio={fail_ratio:.2f}")
                write_slo_violation()
                requests.post(alarm_url, json=alarm_body, headers=alarm_headers)
            else:
                print(f"[OK] latency={avg_lat:.3f}s, fail={fail}/{total}")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
