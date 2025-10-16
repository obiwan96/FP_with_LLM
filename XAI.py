import os, json, time, requests, shap, torch
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import classification_report
import joblib

# ==== 경로/엔드포인트 설정 ====
MODEL_PATH = "model.pt"
SCALER_PATH = "scaler.joblib"
FEAT_PATH   = "feature_names.csv"
BG_PATH     = "bg_samples.npy"

LOKI_URL = "http://<loki-host>:3100/loki/api/v1/query_range"
NS = "oai"; SMF_CONTAINER = "smf"  # 필요시 바꿔쓰기
OLLAMA_URL = "http://localhost:11434/api/generate"  # 기본 포트

TOPK = 8      # SHAP 상위 피처 개수
TOLERANCE_S = 30
LLM_MODEL = "llama3.1"

# ==== 1) 모델/스케일러/특징 로딩 ====
from torch import nn
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, x):
        out,_ = self.lstm(x)
        return self.fc(out[:,-1,:])

# 준비물 로딩
scaler = joblib.load(SCALER_PATH)
feature_names = pd.read_csv(FEAT_PATH, header=None)[0].tolist()
input_size = len(feature_names)

model = LSTMModel(input_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# 확률 반환 래퍼 (SHAP용)
@torch.no_grad()
def predict_proba(x_np: np.ndarray) -> np.ndarray:
    """x_np shape: (N, T, F)  -> returns probability for class 1"""
    x = torch.tensor(x_np, dtype=torch.float32)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
    # KernelExplainer는 다중 출력도 받으니 2클래스 확률로 반환해도 됨
    return np.stack([1-prob, prob], axis=1)

# ==== 2) 테스트 데이터(X_test, y_test, timestamps_test) 로드(예시) ====
# 이미 학습 파이프라인에서 저장해 두었다고 가정
X_test = np.load("X_test.npy")           # shape: (N, T, F)  (스케일 적용 후)
y_test = np.load("y_test.npy")           # shape: (N,)
ts_test = pd.to_datetime(pd.read_csv("timestamps_test.csv")["ts"])  # 각 샘플의 라벨 시점

# ==== 3) 고장으로 예측된 샘플 인덱스 선택 ====
pred_prob = predict_proba(X_test)[:,1]
y_pred = (pred_prob >= 0.5).astype(int)
print(classification_report(y_test, y_pred, digits=3))

fail_idx = np.where(y_pred == 1)[0]
if len(fail_idx) == 0:
    print("No predicted failures in test set."); exit(0)

# ==== 4) SHAP 계산 (빠른 KernelExplainer / 샘플링) ====
# 배경 샘플: 학습셋 일부 or 테스트셋 일부
bg = np.load(BG_PATH)                    # shape: (B, T, F)
# 설명 대상: 예측=고장 인 샘플 중 상위 K개만
K = min(10, len(fail_idx))
target_samples = X_test[fail_idx[:K]]

# KernelExplainer는 2D 입력을 기대 → (N, T*F) 로 펼쳐서 사용
def flatten_TS(x3d):  # (N,T,F)->(N,T*F)
    N,T,F = x3d.shape
    return x3d.reshape(N, T*F)
def unflatten_TS(x2d, T, F):  # (N,T*F)->(N,T,F)
    N = x2d.shape[0]
    return x2d.reshape(N, T, F)

T = X_test.shape[1]; F = X_test.shape[2]
bg_flat  = flatten_TS(bg)
tar_flat = flatten_TS(target_samples)

# 예측함수도 2D 입력을 받을 수 있도록 래핑
def predict_proba_flat(x2d):
    x3d = unflatten_TS(x2d, T, F)
    return predict_proba(x3d)

explainer = shap.KernelExplainer(predict_proba_flat, bg_flat[:100])  # 100개 정도로 제한
shap_vals = explainer.shap_values(tar_flat, nsamples=100)            # 2-class 출력
# class-1(고장)의 shap 값만 사용
shap_fail = shap_vals[1]              # shape: (K, T*F)
shap_fail_3d = unflatten_TS(shap_fail, T, F)   # (K, T, F)

# 시간축 집계: |SHAP|를 시간에 대해 합산 → 피처(메트릭) 중요도
abs_importance = np.abs(shap_fail_3d).mean(axis=0).sum(axis=0)  # (F,)
feat_importance = pd.Series(abs_importance, index=feature_names).sort_values(ascending=False)
top_feats = feat_importance.head(TOPK)
print("\n[Top SHAP features]\n", top_feats)

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
