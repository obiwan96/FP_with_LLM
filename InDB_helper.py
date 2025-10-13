from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime
from secret import InDB_info

url = InDB_info['url']
token = InDB_info['token']
org = InDB_info['org']
bucket = 'mdaf'

def InDB_write(metric_name, field_name, data, abnormality=False):
    client = InfluxDBClient(url=url, token=token, org=org)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    point = Point(metric_name) \
        .tag("abnormality", 1 if abnormality else 0) \
        .field(field_name, data) \
        .time(datetime.utcnow(), WritePrecision.NS)

    write_api.write(bucket = bucket, org=org, record=point)


def InDB_inquiry(metric_name, field_name, start, end, abnormality = False):
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