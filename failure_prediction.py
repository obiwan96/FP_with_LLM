import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from influxdb_client import InfluxDBClient
from Prome_helper import PrometheusClient, to_rfc3339
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from torch import nn, optim
import torch, gc
from torch.utils.data import DataLoader, TensorDataset
import warnings
from secret import InDB_info, prometheus_ip
import joblib
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.utils import resample
import random
import matplotlib.pyplot as plt
from scipy.stats import skew, ttest_ind
from sklearn.metrics import roc_auc_score, average_precision_score
import os
warnings.filterwarnings("ignore")

''' Usuage
# LSTM + raw feature + pod granularity
python train_failure_predictor.py \
  --model LSTM --feature raw --granularity pod
'''

# ---------- 모델 정의 ----------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, x):
        out, _ = self.gru(x)
        #return self.fc(out[:, -1, :])
        out = out.mean(dim=1)
        return self.fc(out)

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        return attn @ v

class GRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.attn = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, x):
        out, _ = self.gru(x)
        attn_out = self.attn(out)
        return self.fc(attn_out[:, -1, :])

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.net = nn.Sequential(self.conv, self.relu)
    def forward(self, x):
        out = self.net(x)
        return out[:, :, :-self.conv.padding[0]]  # trim causal padding

class TCNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, levels=3):
        super().__init__()
        layers = []
        for i in range(levels):
            in_ch = input_size if i == 0 else hidden_size
            layers.append(TCNBlock(in_ch, hidden_size, dilation=2**i))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x)
        out = out.transpose(1, 2)
        return self.fc(out[:, -1, :])

# ---------- 데이터 로드 ----------
def get_prom_metric(prom, metric_name, start, end, step="1m", ns="oai", granularity="pod_avg"):
    if metric_name.endswith("_total"):
        # Counter metric은 rate() 처리
        query = f'rate({metric_name}{{namespace="{ns}"}}[{step}])'
    else:
        # Gauge metric은 raw 값 그대로
        query = f'{metric_name}{{namespace="{ns}"}}'
    data = prom.query_range(query, start=start, end=end, step=step)['data']['result']
    if not data:
        return pd.DataFrame(columns=["timestamp", metric_name])
    rows = []
    for item in data:
        pod = item["metric"].get("pod", "unknown")
        container = item["metric"].get("container", "unknown")
        for ts, val in item["values"]:
            rows.append({
                "timestamp": pd.to_datetime(float(ts), unit="s"),
                "pod": pod,
                "container": container,
                metric_name: float(val)
            })

    df = pd.DataFrame(rows)
    df = df.copy()
    # 정렬 및 타입 정리
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    if granularity == "pod_avg":
        return df.groupby("timestamp")[metric_name].mean().reset_index().sort_values("timestamp")
    elif granularity == "pod":
        return df.groupby(["timestamp", "pod"])[metric_name].mean().reset_index().sort_values("timestamp")
    elif granularity == "node":
        # ToDo
        # node granularity는 Prometheus label 변경 필요 (예시)
        return df.groupby("timestamp")[f"{metric_name}_{pod}"].mean(axis=1).reset_index()
    return df

def get_influx_metrics(influx, start, end, bucket='mdaf'):
    query = f'''
    from(bucket: "{bucket}")
      |> range(start: {start}, stop: {end})
      |> filter(fn: (r) => r["_measurement"] == "core" or r["_measurement"] == "ran")
      |> filter(fn: (r) => r["_field"] != "pdu_session_delay")
      |> toFloat()
      |> group(columns: []) 
      |> keep(columns: ["_time", "_field", "_value"])
    '''
    df = influx.query_api().query_data_frame(query)

    if df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"_time": "timestamp", "_value": "value", "_field": "field"})
    df = df.pivot_table(index="timestamp", columns=["field"], values="value")
    df = df.reset_index()
    return df

def get_failure_and_recovery(influx, start, end, bucket="mdaf"):
    """
    Influx에서 failure_history와 recovery_timestamp를 모두 읽어 반환.
    """
    query = f'''
    from(bucket: "{bucket}")
      |> range(start: {start}, stop: {end})
      |> filter(fn: (r) => r["_measurement"] == "failure_history")
      |> filter(fn: (r) => r["_field"] == "failure_history" or r["_field"] == "recovery_timestamp")
      |> filter(fn: (r) => r["_value"] != "slo violation")
      |> keep(columns: ["_time", "_field", "_value"])
    '''
    tables = influx.query_api().query_data_frame(query)
    if tables.empty:
        print("[WARN] No failure/recovery data found.")
        return pd.DataFrame(columns=["timestamp", "type"])
    df = tables[["_time", "_field", "_value"]].rename(columns={"_time": "timestamp", "_field": "type", "_value": "label"})
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["type"] = df["type"].replace({"failure_history": "failure", "recovery_timestamp": "recovery"})
    return df.sort_values("timestamp").reset_index(drop=True)

def get_slo_violation_history(influx, start, end, bucket='mdaf'):
    query = f'''
    from(bucket: "{bucket}")
      |> range(start: {start}, stop: {end})
      |> filter(fn: (r) => r["_measurement"] == "failure_history")
      |> filter(fn: (r) => r["_field"] == "failure_history")
      |> filter(fn: (r) => r["_value"] == "slo violation")
      |> keep(columns: ["_time", "_value"])
    '''
    tables = influx.query_api().query_data_frame(query)
    print(f'[INFO] total {len(tables)} number of SLO violoation read.')
    if not tables.empty:
        df = tables[["_time", "_value"]].rename(columns={"_time": "timestamp", "_value": "label"})
        df["label"] = 1
        return df
    
    return pd.DataFrame(columns=["timestamp", "label"])

def make_dataset(X, y, window, horizon):
    xs, ys = [], []
    for i in range(len(X) - window - horizon):
        xs.append(X[i : i + window])           # 과거 구간
        future_window = y[i + window : i + window + horizon]
        label = 1 if np.any(future_window > 0) else 0 
        ys.append(label)
    return np.array(xs), np.array(ys)

def analyze_features_cli(X, feature_names=None, output_dir="tmp/", skew_threshold=10):
    """
    각 feature별 평균, 표준편차, 분산, 왜도(skewness)를 계산하고
    log 변환이 필요한 feature를 추천합니다.
    CLI 환경에서도 작동하도록 /tmp 폴더에 히스토그램 이미지를 저장합니다.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        입력 데이터 (shape: [samples, features])
    feature_names : list or None
        feature 이름. None이면 f0, f1, ... 자동 생성
    output_dir : str
        그래프 저장 디렉토리 (기본: /tmp)
    skew_threshold : float
        절댓값이 이 값을 넘는 경우 log 변환 추천
    """
    # DataFrame 변환
    X=X.fillna(0)
    if isinstance(X, np.ndarray):
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
    else:
        df = X.copy()
        feature_names = df.columns

    # 통계량 계산
    stats = pd.DataFrame({
        "mean": df.mean(),
        "std": df.std(),
        "var": df.var(),
        "skewness": df.apply(skew)
    })
    
    stats["recommend_log"] = stats["skewness"].abs() > skew_threshold

    print("\n📊 Feature Statistics Summary\n")
    print(stats.round(4))
    print("\n💡 Log transform recommended for these features:")
    print(stats[stats["recommend_log"]].index.tolist())

    # 시각화 저장
    print(f"\n📁 Saving feature histograms to {output_dir}/ ...")
    for col in feature_names:
        plt.figure(figsize=(5, 3))
        plt.hist(df[col].dropna(), bins=40, color="steelblue", alpha=0.7)
        plt.title(f"{col}\nmean={df[col].mean():.2f}, skew={skew(df[col]):.2f}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()

        # 파일 경로
        save_path = os.path.join(output_dir, f"{col}_hist.png")
        plt.savefig(save_path)
        plt.close()

    print("✅ All histograms saved successfully.\n")

    return stats[stats["recommend_log"]].index.tolist()

def analyze_feature_shift(xs, ys, feature_names):
    """
    기능:
      horizon 이후에 abnormal(=1)이 발생한 윈도우 vs 그렇지 않은 윈도우의
      feature별 평균, 표준편차, 분산 비교표 출력
    """
    xs = np.array(xs)  # shape [N, window, D]
    ys = np.array(ys)  # shape [N,]

    # 윈도우 내 평균값 (시간축 평균)
    X_window_mean = xs.mean(axis=1)  # shape [N, D]

    # normal / abnormal 그룹 분리
    X_normal = X_window_mean[ys == 0]
    X_abnormal = X_window_mean[ys == 1]

    # 통계 요약표 생성
    stats_df = pd.DataFrame({
        "normal_mean": X_normal.mean(axis=0),
        "abnormal_mean": X_abnormal.mean(axis=0),
        "normal_std": X_normal.std(axis=0),
        "abnormal_std": X_abnormal.std(axis=0),
        "mean_diff": X_abnormal.mean(axis=0) - X_normal.mean(axis=0),
        "p_value": ttest_ind(X_abnormal, X_normal, equal_var=False)[1]
    }, index=feature_names)
    
    print("\n📊 Feature Statistics Comparison (normal vs abnormal)\n")
    print(stats_df)

    # P-value 가 작은 feature 상위 3개 표시
    print("\n🔥 Top features with smallest P-value:")
    print(stats_df["p_value"].abs().sort_values(ascending=True).head(3))
    return stats_df

# ---------- 메인 ----------
def main(args):
    print(f"\n[INFO] Starting training with model={args.model}, granularity={args.granularity}, feature={args.feature}\n")

    prom = PrometheusClient(prometheus_ip)
    influx = InfluxDBClient(InDB_info['url'], token=InDB_info['token'], org=InDB_info['org'])

    metrics = {
        "cpu": "container_cpu_usage_seconds_total",
        "mem": "container_memory_usage_bytes",
        "net": "container_network_transmit_bytes_total",
        "disk": "container_fs_writes_bytes_total"
        #"disk2": "container_fs_writes_total"
    }
    if args.end is not None:
        end = to_rfc3339(datetime.fromisoformat(args.end.replace("Z", "+00:00")))
    else:
        end = to_rfc3339()
    start = to_rfc3339(datetime.fromisoformat(args.start.replace("Z", "+00:00")))
    
    prom_df = [get_prom_metric(prom, m, start, end, step= args.step, granularity=args.granularity).sort_values("timestamp") for m in metrics.values()]
    merged = prom_df[0]
    for df in prom_df[1:]:
        merged = pd.merge_asof(merged.sort_values("timestamp"), df.sort_values("timestamp"), on="timestamp", direction="nearest")
    #print(merged.columns)
    influx_df = get_influx_metrics(influx, start, end)

    # Timestamp 정렬 및 병합
    influx_df = influx_df.sort_values("timestamp")
    merged["timestamp"] = pd.to_datetime(merged["timestamp"]).dt.tz_localize(None)
    influx_df["timestamp"] = pd.to_datetime(influx_df["timestamp"]).dt.tz_localize(None)
    merged = pd.merge_asof(merged, influx_df, on="timestamp", direction="nearest", tolerance=pd.Timedelta(args.step))

    merged["timestamp"] = pd.to_datetime(merged["timestamp"]).dt.tz_localize(None)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # === 고장/회복 데이터 읽기 ===
    events_df = get_failure_and_recovery(influx, start, end)

    # === failure-recovery 구간 식별 ===
    failures = events_df[events_df["type"] == "failure"]["timestamp"].tolist()
    recoveries = events_df[events_df["type"] == "recovery"]["timestamp"].tolist()

    # recovery가 없을 경우 대비
    if not failures:
        print("[INFO] No failure events found.")
    elif not recoveries:
        print("[WARN] No recovery events found. Keeping all data post-failure.")
    print(f"[INFO] total {len(failures)} num. of failures read")
    print(f"[INFO] total {len(recoveries)} num. of recovery point read")
    drop_ranges = []
    for f_time in failures:
        # f_time 이후의 가장 가까운 recovery_time 찾기
        rec_after = [r for r in recoveries if r > f_time]
        if rec_after:
            r_time = min(rec_after)
        else:
            # recovery가 없는 마지막 고장은 데이터 끝까지 제거
            r_time = merged["timestamp"].max()
        drop_ranges.append((f_time, r_time))

    '''
    print("[INFO] Excluding failure→recovery intervals:")
    for f, r in drop_ranges:
        print(f"  {f}  →  {r}")
    '''

    # === merged에서 해당 구간 제거 ===
    mask = pd.Series(False, index=merged.index)
    for f, r in drop_ranges:
        mask |= (merged["timestamp"] >= f) & (merged["timestamp"] <= r)

    before = len(merged)
    merged = merged.loc[~mask].reset_index(drop=True)
    after = len(merged)
    print(f"[INFO] Removed {before - after} rows between failure→recovery intervals.")

    # === 이후 failure label 태깅 (회복된 이후 데이터만 포함) ===
    failure_df = pd.DataFrame({"timestamp": failures})
    failure_df["label"] = 1
    merged = pd.merge_asof(
        merged,
        failure_df,
        on="timestamp",
        direction="forward",
        tolerance=pd.Timedelta(args.step)
    )
    slo_df = get_slo_violation_history(influx, args.start, datetime.now(timezone.utc).isoformat()).sort_values("timestamp")
    slo_df["timestamp"] = pd.to_datetime(slo_df["timestamp"]).dt.tz_localize(None)
    if args.slo_as_input:
        merged["label"] = merged.get("label", 0).fillna(0).astype(int)
        if slo_df.empty:
            merged["slo_violation"] = 0
        else:
            # merged의 각 timestamp에 대해 slo_df 중 (t - step, t] 내 이벤트 수 계산
            slo_counts = []
            slo_times = slo_df["timestamp"].values

            for t in merged["timestamp"]:
                count = ((slo_times > (t - pd.Timedelta(args.step)).to_datetime64()) &
                        (slo_times <= t.to_datetime64())).sum()
                slo_counts.append(count)

            merged["slo_violation"] = slo_counts
    #print(slo_df)
    else:
        # Label 병합
        merged = pd.merge_asof(merged, slo_df, on="timestamp", direction="forward",tolerance=pd.Timedelta(args.step))
        merged["label"] = (
            merged.get("label_x", 0).fillna(0).astype(int) |
            merged.get("label_y", 0).fillna(0).astype(int)
        )

        merged.drop(columns=[col for col in ["label_x", "label_y"] if col in merged.columns], inplace=True)

    print (f'[INFO] Total abnormal row num: {(merged["label"] == 1).sum()}')
    
    # feature transform
    feats = merged.drop(columns=["timestamp", "label"])
    print(f'[INFO] input data shape: {merged.shape}')

    # Remove features that std=0 to prevetn Loss become 0
    df_std = feats.std(axis=0)
    valid_cols = df_std[df_std > 0].index
    removed_cols = df_std[df_std == 0].index 
    print("[INFO] Removed columns with std=0:", list(removed_cols))
    feats = feats[valid_cols]
    feature_names = list(feats.columns)
    
    if args.feature == "diff":
        feats = feats.diff().fillna(0)
    elif args.feature == "var":
        feats = feats.rolling(window=3).var().fillna(0)
    else:
        log_recommended_feature_list= analyze_features_cli(feats)
        for feature_name in log_recommended_feature_list:
            feats[feature_name] = np.log1p(feats[feature_name])
    X = feats
    y = merged["label"].values
    X_seq, y_seq = make_dataset(X, y, args.win, args.hor)
    X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
    analyze_feature_shift(X_seq, y_seq, feature_names)

    split = int(len(X_seq) * args.train_ratio)
    split_bound = random.randrange(0,split)
    test_num= len(X_seq)-split
    X_train, y_train = np.append(X_seq[:split_bound],X_seq[split_bound+test_num:],axis=0), np.append(y_seq[:split_bound],y_seq[split_bound+test_num:],axis=0)
    X_test, y_test = X_seq[split_bound:split_bound+test_num], y_seq[split_bound:split_bound+test_num]
    assert(len(X_train)+len(X_test)==len(X_seq))

    # Normalization based on only train data
    N, T, D = X_train.shape
    # (N*T, D)로 reshape
    X_train_flat = X_train.reshape(-1, D)
    X_test_flat  = X_test.reshape(-1, D)
    scaler = StandardScaler()
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    X_test_scaled_flat  = scaler.transform(X_test_flat)
    X_train = X_train_scaled_flat.reshape(N, T, D)
    X_test  = X_test_scaled_flat.reshape(X_test.shape[0], T, D)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Oversampling
    normal_idx = np.where(y_train == 0)[0]
    abnormal_idx = np.where(y_train == 1)[0]
    n_normal, n_abnormal = len(normal_idx), len(abnormal_idx)
    print(f"[INFO] Train Normal={n_normal}, Abnormal={n_abnormal}")
    normal_idx_test = np.where(y_test == 0)[0]
    abnormal_idx_test = np.where(y_test == 1)[0]
    n_normal_test, n_abnormal_test = len(normal_idx_test), len(abnormal_idx_test)
    print(f"[INFO] Test Normal={n_normal_test}, Abnormal={n_abnormal_test}")
    # Weigth 값은 oversampling 이전에 계산.
    class_counts = np.bincount(y_train.astype(int))
    weights = class_counts.sum() / (2.0 * class_counts)
    # abnormal 데이터를 normal 개수의 70~100% 수준까지 복제 (조정 가능)
    target_abn = int(min(n_normal * 0.6, n_abnormal * 10))
    if target_abn > n_abnormal:
        X_abn_resampled, y_abn_resampled = resample(
            X_train[abnormal_idx],
            y_train[abnormal_idx],
            replace=True,
            n_samples=target_abn,
            random_state=42
        )
        X_train_bal = np.concatenate([X_train[normal_idx], X_abn_resampled])
        y_train_bal = np.concatenate([y_train[normal_idx], y_abn_resampled])
    else:
        X_train_bal, y_train_bal = X_train, y_train
    print(f"[INFO] After oversampling Train abnormal: {np.bincount(y_train_bal.astype(int))}")
    print(f"Train data shape: {X_train_bal.shape}, {y_train_bal.shape}")
    X_train_t = torch.tensor(X_train_bal, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_bal, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    #X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    #X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=args.batch, num_workers=0)

    input_size = X_train.shape[2]
    if args.model == "LSTM":
        model = LSTMModel(input_size)
    elif args.model == "GRU":
        model = GRUModel(input_size)
    elif args.model == "GRU_Att":
        model = GRUWithAttention(input_size)
    #elif args.model == "TCN":
    #    model = TCNModel(input_size)
    else:
        model = LSTMModel(input_size)
    weights = torch.tensor(weights, dtype=torch.float32)
    print(f"[INFO] Class weights:", weights.tolist())
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch%10 == 9:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    out = model(xb)
                    preds.extend(torch.argmax(out, 1).numpy())
                    trues.extend(yb.numpy())
            train_f1 = f1_score(trues, preds, average="binary")
            print(f"[Epoch {epoch+1:02d}] Loss={epoch_loss/len(train_loader):.4f}, F1={train_f1:.4f}")

            print("[INFO] Saving SHAP-related resources...")
            torch.save(model.state_dict(), f"tmp/models/{args.model}_model_win{args.win}_hor{args.hor}_{int(train_f1*100)}.pt")               # 학습된 가중치
            joblib.dump(scaler, f"tmp/models/{args.model}_win{args.win}_hor{args.hor}_{int(train_f1*100)}_scaler.joblib")                     # normalization 객체
            pd.Series(feature_names).to_csv("feature_names.csv", index=False)  # feature 이름
            np.save(f"tmp/models/{args.model}_win{args.win}_hor{args.hor}_{int(train_f1*100)}_bg_samples.npy", X_train_bal[:512])             # 학습셋 일부 (SHAP background)
            #print("[INFO] Saved: model.pt, scaler.joblib, feature_names.csv, bg_samples.npy")

    # evaluation
    model.eval()
    preds, trues = [], []
    probs_val, y_val =[], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            preds.extend(torch.argmax(out, 1).numpy())
            trues.extend(yb.numpy())

        # Threshold 최적화
        for xb, yb in train_loader:
            probs_train = torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
            #out = model(xb)
            #probs_val.extend(torch.argmax(out, 1).numpy())
            probs_val.extend(probs_train)
            y_val.extend(yb.numpy())
    p, r, th  = precision_recall_curve(y_val, probs_val)
    f1 = (2*p*r/(p+r+1e-12))
    print(th)
    best = th[np.argmax(f1[:-1])]  # 마지막 th는 정의 상 제외
    print("best_th:", best)

    # 테스트 적용
    pred_test = (preds >= best).astype(int)
    print("F1_test:", f1_score(trues, pred_test))
    test_f1 = f1_score(trues, preds, average="binary")
    print("ROC-AUC:", roc_auc_score(y_test, preds))
    print("PR-AUC:", average_precision_score(y_test, preds))
    print("\n[RESULT] Classification Report:")    
    print(f"\n[FINAL TEST] F1-score = {test_f1:.4f}")
    print(classification_report(trues, preds, digits=3))
    print(f"Model saved in 'tmp/models/{args.model}_model_win{args.win}_hor{args.hor}_{int(train_f1*100)}.pt'.")
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2025-10-14T10:00:00Z")
    parser.add_argument("--end", type=str)
    parser.add_argument("--win", type=int, default=10)
    parser.add_argument("--hor", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--step", type=str, default='1m', help="step for Prometheus, e.g. 1m, 30s")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--granularity", type=str, choices=["pod", "node", "pod_avg"], default="pod_avg")
    parser.add_argument("--feature", type=str, choices=["raw", "diff", "var"], default="raw")
    parser.add_argument("--model", type=str, choices=["LSTM", "GRU", "GRU_Att"], default="GRU")
    parser.add_argument("--slo-as-input", action='store_true')
    args = parser.parse_args()
    main(args)
