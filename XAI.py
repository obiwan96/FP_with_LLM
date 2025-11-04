import os, json, time, requests, torch
from secret import ollama_ip, loki_ip, InDB_info
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import classification_report
import joblib
import argparse
import shap
import re
from learning_helper import LSTMModel, GRUModel, GRUWithAttention, ConvGRU, analyze_features_cli
from InDB_helper import *
from Prome_helper import *
from failure_prediction import get_merged_data, get_slo_violation_history, put_slo_violation_as_input, get_failure_and_recovery
from torch import nn

def find_best_model(model_name, base_dir="tmp/models", restrict_file_path=None):
    # f1 점수가 정수형일 때 (예: 91)
    pattern = re.compile(
        rf"{model_name}_model_win(\d+)_hor(\d+)_([0-9]+)\.pt"
    )

    best_file = None
    best_f1 = -float('inf')
    best_win, best_hor = None, None
    file_list = os.listdir(base_dir)
    if restrict_file_path:
        file_list = [restrict_file_path]
    for filename in file_list:
        match = pattern.match(filename)
        if match:
            win, hor, f1 = match.groups()
            win, hor, f1 = int(win), int(hor), int(f1)

            if f1 > best_f1:
                best_f1 = f1
                best_file = filename
                best_win, best_hor = win, hor

    if best_file:
        return {
            "file": os.path.join(base_dir, best_file),
            "window": best_win,
            "horizon": best_hor,
            "f1": best_f1
        }
    else:
        return None
    
class WrappedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        self.base_model.train()
        out = self.base_model(x)
        if out.dim() == 1:  # (batch,) 형태라면
            out = out.unsqueeze(-1)
        return out

def main(args):
    step_size = 2

    # Read fault history based on end, start date
    if args.end is not None:
        end = to_rfc3339(datetime.fromisoformat(args.end.replace("Z", "+00:00")))
    else:
        end = to_rfc3339()
    start = to_rfc3339(datetime.fromisoformat(args.start.replace("Z", "+00:00")))
    events_df = get_merged_data(start, end, str(f'{step_size}m'), args.granularity)
    slo_df = get_slo_violation_history(args.start, datetime.now(timezone.utc).isoformat()).sort_values("timestamp")
    slo_df["timestamp"] = pd.to_datetime(slo_df["timestamp"]).dt.tz_localize(None)
    events_df= put_slo_violation_as_input(events_df, slo_df, str(f'{step_size}m')) # use slo_violation as input!

    # remove column that std is 0, use log to some column
    df_std = events_df.drop(columns=['timestamp']).std(axis=0)
    valid_cols = df_std[df_std > 0].index
    removed_cols = df_std[df_std == 0].index
    print("[INFO] Removed columns with std=0:", list(removed_cols))
    feature_names = list(events_df[list(valid_cols)].columns)
    events_df = events_df[['timestamp'] + list(valid_cols)]
    num_cols = events_df.columns.difference(['timestamp'])
    feature_df = events_df[num_cols]
    if args.feature == "diff":
        events_df[num_cols] = events_df[num_cols].diff().fillna(0)
    elif args.feature == "var":
        events_df[num_cols] = events_df[num_cols].rolling(window=3).var().fillna(0)
    else:
        log_recommended_feature_list= analyze_features_cli(feature_df)
        for feature_name in log_recommended_feature_list:
            events_df[feature_name] = np.log1p(events_df[feature_name])

    #get failure time stamp list
    failures_recoveries_df = get_failure_and_recovery(start, end)
    #print(failures_recoveries_df.columns)
    failures_df = failures_recoveries_df[failures_recoveries_df["type"] == "failure"]
    failures = list(zip(failures_df["timestamp"], failures_df["label"]))
    #failures = failures_recoveries_df[failures_recoveries_df["type"] == "failure"]["timestamp"].tolist()
    recoveries = failures_recoveries_df[failures_recoveries_df["type"] == "recovery"]["timestamp"].tolist()
    r_time = None
    ori_failure = []
    for f_time, container in failures:
        if r_time and f_time < r_time:
            continue
        print(f'{f_time}, {container}')
        ori_failure.append((f_time, container))
        rec_after = [r for r in recoveries if r > f_time]
        if rec_after:
            r_time = min(rec_after)

    # Find saved best model
    best_model_info = find_best_model(args.model, restrict_file_path=args.restrict)
    if not best_model_info:
        print("Can't find best model.")
        return
    best_model=best_model_info['file']
    window= best_model_info['window']
    horizon = best_model_info['horizon']
    f1 = best_model_info['f1']
    SCALER_PATH = f"tmp/models/{args.model}_model_win{window}_hor{horizon}_{f1}_scaler.joblib"
    FEAT_PATH   = "feature_names.csv"
    BG_PATH     = f"tmp/models/{args.model}_model_win{window}_hor{horizon}_{f1}_bg_samples.npy"
    print(f'[INFO] using model from {best_model}')
    NS = "oai"; SMF_CONTAINER = "smf"  # 필요시 바꿔쓰기

    TOPK = 8      # SHAP 상위 피처 개수
    TOLERANCE_S = 30
    llm_list = ['deepseek-r1:14b', 'gpt-oss:latest',  'gemma3:27b']

    # 준비물 로딩
    scaler = joblib.load(SCALER_PATH)
    #feature_names = pd.read_csv(FEAT_PATH, header=None)[0].tolist()
    input_size = len(feature_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == 'LSTM':
        model = LSTMModel(input_size).to(device)
    elif args.model == 'GRU':
        model = GRUModel(input_size, hidden_size=64).to(device)
    elif args.model == 'GRU_Att':
        model = GRUWithAttention(input_size).to(device)
    else: # CNV_GRU
        model = ConvGRU(input_size).to(device)
    #model = ConvLSTM(input_size).to(device)
    model.load_state_dict(torch.load(best_model, map_location=device))
    bg = np.load(BG_PATH)   
    N, T, D = bg.shape
    bg_2d = bg.reshape(-1, D)  # (N*T, D)

    bg_scaled = scaler.transform(bg_2d)
    bg_scaled = bg_scaled.reshape(N, T, D)                 

    results = []
    for fail_time, container in ori_failure:
        model.eval()
        # 고장 시점 기준 구간 설정
        end_time = fail_time - timedelta(minutes=horizon * step_size)
        start_time = end_time - timedelta(minutes=window * step_size)

        # 해당 구간 데이터 추출
        df_window = events_df[
            (events_df["timestamp"] >= start_time) &
            (events_df["timestamp"] < end_time)
        ]

        if len(df_window) < window:
            print(f"[WARN] {fail_time} 구간 데이터 부족 ({len(df_window)}/{window})")
            continue

        # feature 추출 및 스케일링
        x_input = df_window[feature_names].values
        x_input = np.nan_to_num(x_input, nan=0.0, posinf=0.0, neginf=0.0)
        x_input = scaler.transform(x_input)

        # shape: (1, window, feature_dim)
        x_input = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(device)

        # 모델 예측
        with torch.no_grad():
            pred = torch.sigmoid(model(x_input)).item()

        print(f"[INFO] Failure@{fail_time} at {container} → Pred: {pred:.3f}")
        
        # 예측이 0.5 이상일 경우 SHAP 분석
        if pred >= 0:
            # SHAP용 배경 설정
            background = torch.tensor(
                bg_scaled,
                dtype=torch.float32
            ).to(device)
            model.train()
            wrapped_model = WrappedModel(model)

            explainer = shap.DeepExplainer(wrapped_model, background)
            shap_values = explainer.shap_values(x_input, check_additivity=False)

            # shap_values[0] = (1, window, feature_dim, 1)
            mean_abs_contrib = np.mean(np.abs(shap_values[0]), axis=0).squeeze()
            print(mean_abs_contrib.shape)
            shap_importance = sorted(
                zip(feature_names, mean_abs_contrib),
                key=lambda x: x[1],
                reverse=True
            )

            print(f"[SHAP] Top contributors for {fail_time} at {container}:")
            for feat, val in shap_importance[:5]:
                print(f"   {feat}: {val:.4f}")

            results.append({
                "failure_time": fail_time,
                "pred": pred,
                "container" : container,
                "top_features": shap_importance[:TOPK]
            })


    # ==== 5) ‘어떤 pod’ 문제인지 추론 ====
    # 전제: feature 이름에 pod가 들어있다면(예: 'container_cpu_usage_seconds_total{pod=...}')
    def extract_pod_from_feat(name: str):
        # 예: container_cpu...{pod="oai-upf-xxx", ...}
        if "pod=" in name:
            s = name.split("pod=")[1]
            # 따옴표/괄호 정리
            s = s.split(",")[0].split("}")[0].strip('"{} ')
            return s
        return None

    pod_votes = []
    for feat in top_feats.index:
        pod = extract_pod_from_feat(feat)
        if pod: pod_votes.append(pod)
    pod_rank = pd.Series(pod_votes).value_counts() if pod_votes else pd.Series(dtype=int)
    suspect_pod = pod_rank.index[0] if not pod_rank.empty else None
    print("\n[Suspect pod by SHAP feature names]:", suspect_pod)

    # ==== 6) 해당 pod의 로그를 Loki에서 조회 (예측된 고장 샘플 중 첫 번째 시점 기준, ±5분) ====
    def query_loki(ns, container=None, pod=None, q="|= \"error\" |= \"drop\" |= \"timeout\"", center_ts=None, minutes=5, limit=2000):
        if center_ts is None:
            center_ts = datetime.utcnow()
        start = int((center_ts - timedelta(minutes=minutes)).timestamp() * 1e9)
        end   = int((center_ts + timedelta(minutes=minutes)).timestamp() * 1e9)
        sel = f'{{namespace="{ns}"'
        if container: sel += f', container="{container}"'
        if pod: sel += f', pod="{pod}"'
        sel += "}"
        params = {"query": f'{sel} {q}', "start": start, "end": end, "direction":"FORWARD", "limit": limit}
        r = requests.get(LOKI_URL, params=params); r.raise_for_status()
        out=[]
        for s in r.json().get("data",{}).get("result",[]):
            for ts, line in s["values"]:
                out.append(f'{datetime.fromtimestamp(int(ts)/1e9).isoformat()} {line.strip()}')
        return out[:200]

    center_time = pd.to_datetime(ts_test.iloc[fail_idx[0]])
    logs = query_loki(NS, pod=suspect_pod, q='|~ "(?i)error|drop|timeout|pfcp|5xx|oom"', center_ts=center_time, minutes=5)

    # ==== 7) Ollama LLM에 SHAP+로그를 넣어 “조치안” 질의 ====
    def call_ollama(prompt: str, model=LLM_MODEL, max_tokens=800, temperature=0.2):
        payload = {"model": model, "prompt": prompt, "stream": False, "options":{"num_predict": max_tokens, "temperature": temperature}}
        r = requests.post(OLLAMA_URL, json=payload); r.raise_for_status()
        return r.json().get("response","")

    # 프롬프트 구성 (요약해서 전달)
    feat_text = "\n".join([f"- {k}: {v:.4f}" for k,v in top_feats.items()])
    log_text  = "\n".join(logs[:60])  # 너무 길면 60줄 제한
    sus_line  = f"Suspect pod: {suspect_pod or 'UNKNOWN'}"

    prompt = f"""
    You are an expert SRE for 5G core/cloud-native systems. A time-series ML model predicted an imminent FAILURE.

    Context:
    - {sus_line}
    - Time window center: {center_time.isoformat()}

    Top SHAP features contributing to failure:
    {feat_text}

    Recent logs (OAI SMF/UPF/AMF, Loki, ±5min):
    {log_text}

    Task:
    1) Diagnose the most likely root cause.
    2) Propose concrete remediation steps (ordered, with commands/config hints).
    3) List metrics/logs to watch to confirm recovery.
    Reply in Korean.
    """

    remedy = call_ollama(prompt)
    print("\n[LLM REMEDIATION SUGGESTION]\n", remedy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["LSTM", "GRU", "GRU_Att", "CNV_GRU"], default='CNV_GRU')
    parser.add_argument("--start", type=str, default="2025-10-14T10:00:00Z")
    parser.add_argument("--end", type=str)
    parser.add_argument("--feature", type=str, choices=["raw", "diff", "var"], default="raw")
    parser.add_argument("--granularity", type=str, choices=["pod", "domain", "pod_avg"], default="pod_avg")
    parser.add_argument("--restrict", type=str)
    args = parser.parse_args()
    main(args)