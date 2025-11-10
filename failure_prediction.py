import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from influxdb_client import InfluxDBClient
from Prome_helper import PrometheusClient, to_rfc3339
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from torch import optim
import torch, gc
from torch.utils.data import DataLoader, TensorDataset
import warnings
from secret import InDB_info, prometheus_ip
import joblib
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score
import os
from learning_helper import *
import pickle as pkl
warnings.filterwarnings("ignore")

''' Usuage
# LSTM + raw feature + pod granularity
python train_failure_predictor.py \
  --model LSTM --feature raw --granularity pod
'''

# ---------- 데이터 로드 ----------
def get_prom_metric(prom, metric_name, start, end, step="1m", ns="oai", granularity="pod_avg"):
    def extract_function_name(pod_name: str) -> str:
        """
        Deployment 파드 이름 패턴: <deploy>-<template-hash>-<suffix>
        예) oai-amf-5bbb74657d-rp9h7  -> oai-amf
            5gc-mysql-5d84db7f84-gjwvt -> 5gc-mysql
            ueransim-gnb-5bd5958656-fncmb -> ueransim-gnb
        길이가 3 미만이면 원본 반환.
        """
        if not pod_name:
            return "unknown"
        parts = pod_name.split("-")
        if len(parts) >= 3:
            return "-".join(parts[:-2])
        return pod_name
    if granularity == "domain":
        ns_list = ["oai", "application"]
        ns_selector = f'{{namespace=~"{("|").join(ns_list)}"}}'
    else:
        ns_selector = f'{{namespace="{ns}"}}'
    if metric_name.endswith("_total"):
        # Counter metric은 rate() 처리
        query = f'rate({metric_name}{ns_selector}[{step}])'
    else:
        # Gauge metric은 raw 값 그대로
        query = f'{metric_name}{ns_selector}'
    data = prom.query_range(query, start=start, end=end, step=step)['data']['result']
    if not data:
        # granularity에 맞는 최소 컬럼만 반환
        if granularity == "pod":
            return pd.DataFrame(columns=["timestamp"])
        elif granularity == "domain":
            return pd.DataFrame(columns=["timestamp"])
        else:  # pod_avg
            return pd.DataFrame(columns=["timestamp", f"{metric_name}_avg"])
    rows = []
    for item in data:
        m = item.get("metric", {})
        pod = m.get("pod", "unknown")
        container = m.get("container", "unknown")
        namespace = m.get("namespace", "unknown")
        func = extract_function_name(pod)
        for ts, val in item["values"]:
            rows.append({
                "timestamp": pd.to_datetime(float(ts), unit="s"),
                "pod": pod,
                "function": func,
                "container": container,
                "namespace": namespace,
                metric_name: float(val)
            })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["timestamp"])

    if granularity == "pod_avg":
        out = (
            df.groupby("timestamp", as_index=False)[metric_name]
              .mean()
              .sort_values("timestamp")
              .rename(columns={metric_name: f"{metric_name}_avg"})
        )
        return out

    elif granularity == "pod":
        # 같은 function(배포 단위) 기준으로 시점별 평균 → wide 피벗
        # 한 시점에 롤링 중이라 두 파드가 있으면 평균 1개 값으로 수렴
        agg = (
            df.groupby(["timestamp", "function"], as_index=False)[metric_name]
              .mean()
        )
        wide = (
            agg.pivot_table(index="timestamp",
                            columns="function",
                            values=metric_name)
               .sort_index()
        )
        # 컬럼명: {metric_name}_{function}
        wide.columns = [f"{metric_name}_{fn}" for fn in wide.columns]
        wide = wide.reset_index()
        return wide

    elif granularity == "domain":
        # namespace별 평균 → wide 피벗
        agg = (
            df.groupby(["timestamp", "namespace"], as_index=False)[metric_name]
              .mean()
        )
        wide = (
            agg.pivot_table(index="timestamp",
                            columns="namespace",
                            values=metric_name)
               .sort_index()
        )
        # 컬럼명: {metric_name}_{ns}
        wide.columns = [f"{metric_name}_{ns_col}" for ns_col in wide.columns]
        wide = wide.reset_index()
        return wide
    # 기본: 원본
    return df

def get_influx_metrics(start, end, bucket='mdaf'):
    influx = InfluxDBClient(InDB_info['url'], token=InDB_info['token'], org=InDB_info['org'])
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

def get_failure_and_recovery(start, end, bucket="mdaf"):
    """
    Influx에서 failure_history와 recovery_timestamp를 모두 읽어 반환.
    """
    influx = InfluxDBClient(InDB_info['url'], token=InDB_info['token'], org=InDB_info['org'])
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

def get_slo_violation_history(start, end, bucket='mdaf'):
    influx = InfluxDBClient(InDB_info['url'], token=InDB_info['token'], org=InDB_info['org'])
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

def put_slo_violation_as_input(merged, slo_df, step, if_failure_data_exist=False):
    if if_failure_data_exist:
        merged["label"] = merged.get("label", 0).fillna(0).astype(int)
    if slo_df.empty:
        merged["slo_violation"] = 0
    else:
        # merged의 각 timestamp에 대해 slo_df 중 (t - step, t] 내 이벤트 수 계산
        slo_counts = []
        slo_times = slo_df["timestamp"].values

        for t in merged["timestamp"]:
            count = ((slo_times > (t - pd.Timedelta(step)).to_datetime64()) &
                    (slo_times <= t.to_datetime64())).sum()
            slo_counts.append(count)

        merged["slo_violation"] = slo_counts
    return merged

def get_prometheus_data(start, end, step, granularity):
    metrics = {
        "cpu": "container_cpu_usage_seconds_total",
        "mem": "container_memory_usage_bytes",
        "net": "container_network_transmit_bytes_total",
        "net2": "container_network_receive_bytes_total",
        "disk": "container_fs_writes_bytes_total"
        #"disk2": "container_fs_writes_total"
    }
    prom = PrometheusClient(prometheus_ip)
    prom_df = [get_prom_metric(prom, m, start, end, step= step, granularity=granularity).sort_values("timestamp") for m in metrics.values()]
    merged = prom_df[0]
    for df in prom_df[1:]:
        merged = pd.merge_asof(merged.sort_values("timestamp"), df.sort_values("timestamp"), on="timestamp", direction="nearest")
    #print(merged.columns)
    return merged

def get_merged_data(start, end, step, granularity, single_domain=None, resource_only=False):
    merged = get_prometheus_data(start, end,step, granularity)
    if resource_only:
        influx_df = pd.DataFrame(columns=["timestamp"])
    else:
        influx_df = get_influx_metrics(start, end)
    if single_domain:
        if single_domain=='ran':
            use_metric_list = ['rrc_state_counts', 'ue_failure_counts']
            nf_list = ['gnb']
        elif single_domain=='core':
            use_metric_list = ["pdu_session_delay_seconds", "amf_registration_rate", "upf_throughput", "smf_session_drop"]
            nf_list = ['amf', 'ausf', 'lmf', 'nrf', 'smf', 'udm', 'udr', 'upf']
        #print(merged.columns)
        merged = merged[['timestamp'] + [col for col in merged.columns if any(nf in col for nf in nf_list)]]
        influx_df = influx_df[['timestamp'] + [col for col in influx_df.columns if col in use_metric_list]]
    # Timestamp 정렬 및 병합
    influx_df = influx_df.sort_values("timestamp")
    merged["timestamp"] = pd.to_datetime(merged["timestamp"]).dt.tz_localize(None)
    influx_df["timestamp"] = pd.to_datetime(influx_df["timestamp"]).dt.tz_localize(None)
    merged = pd.merge_asof(merged, influx_df, on="timestamp", direction="nearest", tolerance=pd.Timedelta(step))

    merged["timestamp"] = pd.to_datetime(merged["timestamp"]).dt.tz_localize(None)
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged

# ---------- 메인 ----------
def main(args):
    print(f"\n[INFO] Starting training with model={args.model}, granularity={args.granularity}, feature={args.feature}\n")
    file_path = f"tmp/data/{args.single_domain}{'resource' if args.resource_only else 'mdaf'}{args.feature}_{args.granularity}_{'sloasinput' if args.slo_as_input else ''}_stepsize{args.step}.pkl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.use_pickle:
        with open(file_path, 'rb') as f:
            dataset= pkl.load(f)
        print(f'📁 Load datset from {file_path}')
        X,y, timestamps, failures = dataset
    else:
        if args.end is not None:
            end = to_rfc3339(datetime.fromisoformat(args.end.replace("Z", "+00:00")))
        else:
            end = to_rfc3339()
        start = to_rfc3339(datetime.fromisoformat(args.start.replace("Z", "+00:00")))
        
        
        # Read X data
        merged = get_merged_data(start, end, args.step, 
                                 args.granularity, single_domain=args.single_domain, resource_only=args.resource_only)

        # === 고장/회복 데이터 읽기 ===
        events_df = get_failure_and_recovery(start, end)

        # === failure-recovery 구간 식별 ===
        failures = events_df[events_df["type"] == "failure"]
        if args.single_domain:
            if args.single_domain=='ran':
                nf_list = ['gnb', 'gnodeb']
            elif args.single_domain=='core':
                nf_list = ['amf', 'ausf', 'lmf', 'nrf', 'smf', 'udm', 'udr', 'upf']
            failures = failures[failures['label'].isin(nf_list)].reset_index(drop=True)
        failures = failures["timestamp"].tolist()
        recoveries = events_df[events_df["type"] == "recovery"]["timestamp"].tolist()
        
        # recovery가 없을 경우 대비
        if not failures:
            print("[INFO] No failure events found.")
        elif not recoveries:
            print("[WARN] No recovery events found. Keeping all data post-failure.")
        print(f"[INFO] total {len(failures)} num. of failures read")
        print(f"[INFO] total {len(recoveries)} num. of recovery point read")
        drop_ranges = []
        r_time = None
        ori_failure = []
        for f_time in failures:
            if r_time and f_time < r_time:
                continue
            ori_failure.append(f_time)
            # f_time 이후의 가장 가까운 recovery_time 찾기
            rec_after = [r for r in recoveries if r > f_time]
            if rec_after:
                r_time = min(rec_after)
            else:
                # recovery가 없는 마지막 고장은 데이터 끝까지 제거
                r_time = merged["timestamp"].max()
            drop_ranges.append((f_time, r_time))
        print("[INFO] Excluding failure→recovery intervals:")
        for f, r in drop_ranges:
            print(f"  {f}  →  {r}")

        # === merged에서 해당 구간 제거 ===
        mask = pd.Series(False, index=merged.index)
        for f, r in drop_ranges:
            mask |= (merged["timestamp"] >= f) & (merged["timestamp"] < r)

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
        slo_df = get_slo_violation_history(args.start, datetime.now(timezone.utc).isoformat()).sort_values("timestamp")
        slo_df["timestamp"] = pd.to_datetime(slo_df["timestamp"]).dt.tz_localize(None)
        if args.slo_as_input:
            if not args.single_domain == 'core' and not args.resource_only:
                merged= put_slo_violation_as_input(merged, slo_df, args.step, True)
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
        timestamps = merged.loc[merged["label"] == 1, "timestamp"]
        # feature transform
        feats = merged.drop(columns=["timestamp", "label"])
        print(f'[INFO] input data shape: {merged.shape}')
        failures = ori_failure
        timestamps = merged["timestamp"].values
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
        X = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        y = merged["label"].values
        dataset = (X,y, timestamps, failures)
        with open(file_path, 'wb') as f:
            pkl.dump(dataset, f)
        print(f'📁 Datset saved in {file_path}')
    if args.optuna:
        if args.model:
            study = study_optuna(X, y, timestamps, failures, device, timeout = args.optuna*60*60, model_name=args.model)
            df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
            df = df[df["state"] == "COMPLETE"]
            best_per_combo = (
                df.loc[df.groupby(["params_window", "params_horizon"])["value"].idxmin()]
                .sort_values("value")
                .reset_index(drop=True)
            )
            print(best_per_combo[
                ["params_window", "params_horizon", "value",
                "params_hidden_size", "params_dropout", "params_lr",
                "params_alpha", "params_gamma", "params_temperature"]
            ])
            return
        else:
            study = study_optuna(X, y, timestamps, failures, device)        
        best_params = study.best_params
        X_seq, y_seq, ts_seq = make_soft_dataset(X, y, timestamps, best_params["window"], best_params["horizon"], mode="linear")
        if best_params["model_name"] == 'LSTM':
            model = LSTMModel(input_size=X_seq.shape[2],
                         hidden_size=best_params["hidden_size"],
                         dropout=best_params["dropout"], 
                         temperature=best_params["temperature"]).to(device)
        elif best_params["model_name"] == 'GRU':
            model = GRUModel(input_size=X_seq.shape[2],
                         hidden_size=best_params["hidden_size"],
                         dropout=best_params["dropout"], 
                         temperature=best_params["temperature"]).to(device)
        elif best_params["model_name"] == 'GRU_Att':
            model = GRUWithAttention(input_size=X_seq.shape[2],
                         hidden_size=best_params["hidden_size"],
                         dropout=best_params["dropout"], 
                         temperature=best_params["temperature"]).to(device)
        else: # CNV_GRU
            model = ConvGRU(input_size=X_seq.shape[2],
                         hidden_size=best_params["hidden_size"],
                         dropout=best_params["dropout"], 
                         temperature=best_params["temperature"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
        criterion = lambda logits, targets: focal_loss_ce(logits, targets,
                                                    alpha=best_params["alpha"],
                                                    gamma=best_params["gamma"])
    else:
        X_seq, y_seq, ts_seq = make_soft_dataset(X, y, timestamps, args.win, args.hor, mode="linear")
        if args.model == 'LSTM':
            model = LSTMModel(input_size=X_seq.shape[2])
        elif args.model == 'GRU':
            model = GRUModel(input_size=X_seq.shape[2])
        elif args.model == 'GRU_Att':
            model = GRUWithAttention(input_size=X_seq.shape[2])
        else: # CNV_GRU
            model = ConvGRU(input_size=X_seq.shape[2])
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        criterion = lambda logits, targets: focal_loss_ce(logits, targets,alpha=0.9)
    plt.figure(figsize=(10,3))
    plt.plot(y, label='original anomaly flag')
    plt.title("Original Label (binary)")
    save_path = os.path.join('tmp/', f"original_y_hist.png")
    plt.savefig(save_path)
    plt.close()
    plt.figure(figsize=(10,3))
    plt.plot(y_seq, label='soft anomaly flag')
    plt.title("soft Label")
    save_path = os.path.join('tmp/', f"soft_y_hist.png")
    plt.savefig(save_path)
    plt.close()
    print("Label stats:", np.mean(y_seq), np.std(y_seq))
    alpha= 1-np.mean(y_seq == 1)
    print("Label nonzero ratio:", 1-alpha)
    #print("Mean label value:", y_seq.mean())
    #print("Mean y over window:", np.mean([np.mean(y_seq[i:i+args.win]) for i in range(len(y_seq)-args.win)]))
    

    X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
    corrs=[]
    for i in range(X_seq.shape[2]):  # feature dimension
        r, _ = pearsonr(X_seq[:, -1, i], y_seq)
        corrs.append(r)
    print(pd.Series(corrs, index=feature_names).sort_values(ascending=False))
    analyze_feature_shift(X_seq, y_seq, feature_names)

    '''split = int(len(X_seq) * args.train_ratio)
    split_bound = random.randrange(0,split)
    split_bound = split-1 # no random select. only last 
    test_num= len(X_seq)-split
    X_train, y_train = np.append(X_seq[:split_bound],X_seq[split_bound+test_num:],axis=0), np.append(y_seq[:split_bound],y_seq[split_bound+test_num:],axis=0)
    X_test, y_test = X_seq[split_bound:split_bound+test_num], y_seq[split_bound:split_bound+test_num]'''
    #y_seq = smooth_labels(y_seq, eps=0.05)
    cutoff_time = get_cutoff_time_by_failure_ratio(failures, train_ratio=args.train_ratio)
    X_train, X_test, y_train, y_test = split_by_cutoff(X_seq, y_seq, ts_seq, cutoff_time)
    assert(len(X_train)+len(X_test)==len(X_seq))
    plt.figure(figsize=(10,2))
    plt.scatter(ts_seq, np.zeros(len(ts_seq)), s=3, c='gray', label='Samples')
    plt.scatter(ori_failure, np.full(len(ori_failure), 0.1), s=3, c='red', label='Failures')  # ✅ 수정된 부분
    plt.scatter(failures, np.full(len(failures), 0.05), s=3, c='orange', label='Used Failures')  # 실제 학습에 사용된 failure
    for rec_time in recoveries:
        plt.axvline(rec_time, color='green', linestyle=':', alpha=0.5)
    plt.axvline(cutoff_time, color='blue', linestyle='--', label='Cutoff')
    plt.legend()
    plt.title("Train/Test Split by Failure Ratio (Soft Dataset)")
    save_path = os.path.join('tmp/', f"cutoff_visualization.png")
    plt.savefig(save_path)
    plt.close()
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
    # don't oversampling now.
    '''target_abn = int(min(n_normal * 0.6, n_abnormal * 10))
    if target_abn > n_abnormal:
        X_abn_resampled, y_abn_resampled = resample(
            X_train[abnormal_idx],
            y_train[abnormal_idx],
            replace=True,
            n_samples=target_abn,
            random_state=42
        )
        X_train = np.concatenate([X_train[normal_idx], X_abn_resampled])
        y_train = np.concatenate([y_train[normal_idx], y_abn_resampled])
    else:
        X_train, y_train = X_train, y_train
    print(f"[INFO] After oversampling Train abnormal: {np.bincount(y_train.astype(int))}")'''
    print(f"Train data shape: {X_train.shape}, {y_train.shape}")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    print(f'[INFO] train data max value: {np.max(X_train)},  min value: {np.min(X_train)}')
    #X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    #X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=args.batch, num_workers=0)

    
    weights = torch.tensor(weights, dtype=torch.float32)
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #criterion = nn.CrossEntropyLoss(weight=weights)
    
    #criterion = nn.BCEWithLogitsLoss()
    # training
    for epoch in range(args.epochs):
        train_and_eval(model, train_loader, test_loader, optimizer, criterion, device, scheduler)

    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = torch.sigmoid(model(xb)).cpu().numpy()
            preds.extend(out)
            trues.extend(yb.numpy())
    preds, trues = np.array(preds), np.array(trues)
    f1 = f1_score((trues > 0.5).astype(int), preds > 0.5)
    auc = roc_auc_score((trues > 0.5).astype(int), preds)
    pr = average_precision_score((trues > 0.5).astype(int), preds)

    # 요약
    print("\n[RESULT] Test Set Performance:")
    print("F1-score: {:.4f}".format(f1))
    print("AUROC: {:.4f}".format(auc))
    print("AUPRC: {:.4f}".format(pr))
    print("[INFO] Saving SHAP-related resources...")
    torch.save(model.state_dict(), f"tmp/models/{args.model}_model_win{args.win}_hor{args.hor}_{int(f1*100)}.pt")               # 학습된 가중치
    joblib.dump(scaler, f"tmp/models/{args.model}_model_win{args.win}_hor{args.hor}_{int(f1*100)}_scaler.joblib")                     # normalization 객체
    pd.Series(feature_names).to_csv("feature_names.csv", index=False)  # feature 이름
    np.save(f"tmp/models/{args.model}_model_win{args.win}_hor{args.hor}_{int(f1*100)}_bg_samples.npy", X_train[:512])             # 학습셋 일부 (SHAP background)
    print(f"Model saved in 'tmp/models/{args.model}_model_win{args.win}_hor{args.hor}_{int(f1*100)}.pt'.")
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2025-10-14T10:00:00Z")
    parser.add_argument("--end", type=str)
    parser.add_argument("--win", type=int, default=10)
    parser.add_argument("--hor", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--step", type=str, default='3m', help="step for Prometheus, e.g. 1m, 30s")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--granularity", type=str, choices=["pod", "domain", "pod_avg"], default="pod")
    parser.add_argument("--feature", type=str, choices=["raw", "diff", "var"], default="raw")
    parser.add_argument("--model", type=str, choices=["LSTM", "GRU", "GRU_Att", "CNV_GRU"])
    parser.add_argument("--slo-as-input", action='store_true')
    parser.add_argument("--optuna", type=int, help='To use Optuna, put limit ime as hours.')
    parser.add_argument("--use-pickle", action='store_true', help="using saved pickle file. if no, save the data file.")
    parser.add_argument("--single-domain", type=str, choices=["ran", "core"])
    parser.add_argument("--resource-only", action='store_true')
    args = parser.parse_args()
    main(args)
