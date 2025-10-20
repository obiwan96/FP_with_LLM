from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timedelta, timezone
from secret import InDB_info
import argparse

url = InDB_info['url']
token = InDB_info['token']
org = InDB_info['org']
bucket = 'mdaf'

def InDB_write(metric_name:str, field_name:str, data, abnormality=False, timestamp:str=None):
    client = InfluxDBClient(url=url, token=token, org=org)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    if timestamp == None:
        timestamp=datetime.utcnow()
    else:
        timestamp = datetime.fromisoformat(timestamp)
    point = Point(metric_name) \
        .tag("abnormality", 1 if abnormality else 0) \
        .field(field_name, data) \
        .time(timestamp, WritePrecision.NS)

    write_api.write(bucket = bucket, org=org, record=point)

def InDB_write_recovery_timestamp(timestamp:datetime):
    client = InfluxDBClient(url=url, token=token, org=org)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    point = Point('failure_history') \
        .field('recovery_timestamp', 'recovery') \
        .time(timestamp, WritePrecision.NS)

    write_api.write(bucket = bucket, org=org, record=point)

# InDB delete:
# usuage: InDB_delete('2025-10-14T15:00:00Z','2025-10-14T15:00:00Z', '_measurement="core" AND pod="amf-0"')
def InDB_delete(start, stop, predicate):
    client = InfluxDBClient(url=url, token=token, org=org)
    client.delete_api().delete(start, stop, predicate, bucket=bucket, org=org)

def InDB_inquiry(metric_name:str, field_name:str, start:datetime, end:datetime, abnormality:bool = False):
    client = InfluxDBClient(url=url, token=token, org=org)
    query_api = client.query_api()
    start_str = start.isoformat()
    end_str = end.isoformat()

    query = f'''
    from(bucket: "{bucket}")
    |> range(start: {start_str}, stop: {end_str})
    |> filter(fn: (r) => r._measurement == "{metric_name}" and r._field == "{field_name}" and r._abnormality == {1 if abnormality else 0})
    |> aggregateWindow(every: 1m, fn: mean)
    |> yield(name: "mean")
    '''
    
    '''
    InDB의 FluxTable 객체 사용시
    tables = query_api.query(query)
    tables: List[FluxTable]
    └── FluxTable
          └── records: List[FluxRecord]
                └── record.get_time(), record.get_value(), record.get_field(), record.values["tag이름"]
    for table in tables:
        for record in table.records:
            print(f"{record.get_time()} => {record.get_value()}")
    '''
    df = query_api.query_data_frame(query)
    return df

def InDB_inquiry_now(interval:int=10):
    client = InfluxDBClient(url=url, token=token, org=org)
    query_api = client.query_api()
    now = datetime.now(timezone.utc).isoformat()   # UTC 기준 권장
    start = (datetime.now(timezone.utc) - timedelta(minutes=interval)).isoformat()
    query = f'''
    from(bucket: "{bucket}")
      |> range(start: {start}, stop: {now})
      |> filter(fn: (r) => r._measurement == "ran")
      |> last()
    '''
    tables = query_api.query(query)
    for table in tables:
        for record in table.records:
            print(
                f"metric={record.get_measurement()} "
                f"tag={record.values.get('tag', None)} "
                f"field={record.get_field()} "
                f"value={record.get_value()}"
            )

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=str, help='e.g. 2025-10-14T15:00:00Z, Korean time')
    args= ap.parse_args()
    if args.time:
        # Korean time!
        data_time=datetime.fromisoformat(args.time.replace("Z", "+09:00"))
    else: 
        data_time= datetime.now(timezone.utc)
    InDB_write_recovery_timestamp(data_time)