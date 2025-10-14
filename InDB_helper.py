from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timedelta, timezone
from secret import InDB_info

url = InDB_info['url']
token = InDB_info['token']
org = InDB_info['org']
bucket = 'mdaf'

def InDB_write(metric_name:str, field_name:str, data, abnormality=False):
    client = InfluxDBClient(url=url, token=token, org=org)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    point = Point(metric_name) \
        .tag("abnormality", 1 if abnormality else 0) \
        .field(field_name, data) \
        .time(datetime.utcnow(), WritePrecision.NS)

    write_api.write(bucket = bucket, org=org, record=point)


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