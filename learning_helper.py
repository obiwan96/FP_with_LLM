import optuna
from torch import nn
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
from scipy.stats import skew, ttest_ind
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
from functools import partial
import optuna.visualization as vis
from optuna.pruners import MedianPruner
from GNN import visualize_topology_dynamics, explain_node_contributions, visual_node_embeddings, NODE_NAMES
dropout= 0.3
temperature = 0.6

# ---------- 모델 정의 ----------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=dropout, temperature=temperature):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.temperature = nn.Parameter(torch.tensor(temperature))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1) / self.temperature

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=dropout, temperature=temperature):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.temperature = nn.Parameter(torch.tensor(temperature))
    def forward(self, x):
        out, _ = self.gru(x)
        #return self.fc(out[:, -1, :])
        out = out.mean(dim=1)
        return self.fc(out).squeeze(-1) / self.temperature

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
    def __init__(self, input_size, hidden_size=256, num_layers=1, dropout=dropout, temperature=temperature):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attn = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.temperature = nn.Parameter(torch.tensor(temperature))
    def forward(self, x):
        out, _ = self.gru(x)
        attn_out = self.attn(out)
        return self.fc(attn_out[:, -1, :]).squeeze(-1) / self.temperature
    
class ConvGRU(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=1, dropout=dropout, temperature=temperature):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, input_size, kernel_size=5, padding=1)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x):
        # x: [B, T, D] → conv expects [B, D, T]
        x = x.permute(0, 2, 1)
        x = self.conv1(x).permute(0, 2, 1)
        out, _ = self.gru(x)
        out = out.mean(dim=1)
        return self.fc(out).squeeze(-1) / self.temperature

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
    
def focal_loss_ce(logits, targets, alpha=0.9, gamma=1.0, reduction='mean'):
    #ce = F.binary_cross_entropy_with_logits(logits.squeeze(), targets.float(), reduction='none')
    ce = F.binary_cross_entropy_with_logits(logits.view(-1), targets.float().view(-1), reduction='none')

    pt = torch.exp(-ce)
    loss = alpha * ((1 - pt) ** gamma) * ce
    return loss.mean()

##########
## Dataset 관련

def get_cutoff_time_by_failure_ratio(failures, train_ratio=0.8):
    """
    failures 리스트로부터 train_ratio 비율만큼 train에 포함되도록 cutoff 시점 계산.
    """
    failures = pd.to_datetime(failures)
    failures_sorted = np.sort(failures)
    if len(failures_sorted) == 0:
        raise ValueError("⚠️ failures list is empty")

    cutoff_idx = int(np.ceil(len(failures_sorted) * train_ratio)) - 1
    cutoff_idx = max(0, min(cutoff_idx, len(failures_sorted) - 1))
    cutoff_time = failures_sorted[cutoff_idx]

    #print(f"[INFO] Cutoff time = {cutoff_time}")
    #print(f"[INFO] Train failures: {cutoff_idx + 1}/{len(failures_sorted)} "  f"({(cutoff_idx + 1)/len(failures_sorted):.1%})")
    return cutoff_time

def split_by_cutoff(X_seq, y_seq, ts_seq, cutoff_time, put_cutoff_to_train=True):
    """
    cutoff 시점을 기준으로 X/y/timestamp split.
    """
    ts_seq = pd.to_datetime(ts_seq)
    if put_cutoff_to_train:
        train_mask = ts_seq <= cutoff_time
        test_mask = ts_seq > cutoff_time
    else:
        train_mask = ts_seq < cutoff_time
        test_mask = ts_seq >= cutoff_time

    X_train, X_test = X_seq[train_mask], X_seq[test_mask]
    y_train, y_test = y_seq[train_mask], y_seq[test_mask]

    #print(f"[INFO] Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    #print(f"[INFO] Train positives: {np.sum(y_train==1)} | Test positives: {np.sum(y_test==1)}")
    return X_train, X_test, y_train, y_test


def event_cv_split(failures, n_folds=3):
    failures = sorted(failures)
    folds = []
    fold_size = max(1, len(failures) // n_folds)
    for i in range(n_folds):
        val = failures[i*fold_size:(i+1)*fold_size]
        train = [f for f in failures if f not in val]
        if val:
            folds.append((train, val))
    return folds

def make_dataset(X, y, window, horizon):
    xs, ys = [], []
    for i in range(len(X) - window - horizon):
        xs.append(X[i : i + window])           # 과거 구간
        future_window = y[i + window -1 : i + window + horizon-1]
        label = 1 if np.any(future_window > 0) else 0 
        ys.append(label)
    return np.array(xs), np.array(ys)

def make_soft_dataset(X, y, timestamps, window, horizon, smooth_window=3, mode="linear"):
    """
    window/horizon 기반 soft label 데이터셋 생성 + 각 시퀀스의 대표 timestamp 저장
    """
    xs, ys, ts_valid = [], [], []
    N = len(X)
    for i in range(N - window - horizon+1):
        x_window = X[i : i + window]
        if horizon > 0:
            current_y_window = y[i:i + window]
            if np.any(current_y_window > 0):
                continue  # if not detection, fault inside window → skip

        future_window = y[i + window-1: i + window + horizon]
        label = 1 if np.any(future_window > 0) else 0

        if not label:
            future_y = y[i + window + horizon : i + window + horizon + smooth_window]
            if np.any(future_y > 0):
                dist = np.argmax(future_y > 0)
                if mode == "linear":
                    #label = max(0.0, 1.0 - dist / horizon)
                    label = max(0, 1 - dist / (smooth_window))
                elif mode == "exp":
                    label = np.exp(-dist / (smooth_window))
        xs.append(x_window)
        ys.append(label)
        ts_valid.append(timestamps[i + window + horizon - 1])

    return np.array(xs), np.array(ys), np.array(ts_valid)

def smooth_labels(y, eps=0.05):
    """
    Label smoothing for imbalanced soft targets.
    y: numpy array or torch tensor (0~1)
    eps: smoothing factor (default 0.05 → 5% 부드럽게)
    """
    y = np.asarray(y, dtype=np.float32)
    y_smooth = y * (1 - eps) + eps * 0.5
    return y_smooth

# feature 분석/ 학습 관련
def train_and_eval(model, train_loader, val_loader, optimizer, criterion, device, scheduler=None, adj=None, epoch_num=None, skip_eval=False):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        if adj is not None and getattr(model, "requires_graph", False):
            out = model(xb, adj)
        else:
            out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
    if scheduler:
        scheduler.step(loss.item())
    emb_last_step = None
    if epoch_num is not None and epoch_num % 20 == 0 and adj is not None and getattr(model, "requires_graph", False):
        sample_batch = next(iter(train_loader))
        sample_xb = sample_batch[0][:1].numpy() # [1, time, nodes, features] 크기로 슬라이싱
        
        # 1) Saliency Map 기반 노드 중요도 계산
        scores = explain_node_contributions(model, sample_xb, adj, NODE_NAMES, device)
        
        # 2) Pyvis를 활용한 토폴로지 시각화 파일 저장
        visualize_topology_dynamics(
            adj_matrix=adj, 
            node_scores=scores, 
            node_names=NODE_NAMES, 
            filename=f"tmp/gnn/topology_epoch_{epoch_num}.html"
        )
        
        # 3) (선택) 노드 임베딩 공간 분포 시각화 (t-SNE)
        emb_last_step = visual_node_embeddings(model, sample_xb, adj, NODE_NAMES, epoch_num, device)

    if skip_eval:
        return 0, 0, 0, 0, emb_last_step
    # --- Validation ---
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            if adj is not None and getattr(model, "requires_graph", False):
                out = model(xb, adj)
            else:
                out = model(xb)
            preds.extend(torch.sigmoid(out).cpu().numpy())
            trues.extend(yb.cpu().numpy())
    preds, trues = np.array(preds), np.array(trues)
    ths = np.linspace(0.1, 0.9, 9)
    f1s = [f1_score((trues > 0.5).astype(int), (preds > t).astype(int)) for t in ths]
    best_idx = np.argmax(f1s)
    best_th = ths[best_idx]
    f1_best = f1s[best_idx]
    try:
        roc = roc_auc_score((trues > 0.5).astype(int), preds)
        pr = average_precision_score((trues > 0.5).astype(int), preds)
    except Exception as E:
        roc = pr = 0
        print (f'[ERROR] exception ccured in roc, precision score. {E}')
        print(np.unique(trues, return_counts=True))
    return f1_best, roc, pr, best_th, emb_last_step

# ✅ Optuna objective
def objective(trial, X_raw, y_raw, timestamps, failures, device, model_name = None):
    hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    alpha = trial.suggest_float("alpha", 0.6, 1.0)
    gamma = trial.suggest_float("gamma", 0.5, 3.0)
    temperature = trial.suggest_float("temperature", 0.5, 3.0)
    window = trial.suggest_int("window", 5,15, step=5)
    horizon = trial.suggest_categorical("horizon", [0,1,2,3, 5])
    if model_name is None:
        model_name = trial.suggest_categorical("model_name", ["LSTM", "GRU", "GRU_Att", "CNV_GRU"])

    X_seq, y_seq, ts_seq = make_soft_dataset(X_raw, y_raw, timestamps, window, horizon, mode="linear")

    # --- Event-based 2-fold CV ---
    train_ratio_list = [0.3, 0.7]
    fold_scores = []
    for train_ratio in train_ratio_list:
        cutoff_time = get_cutoff_time_by_failure_ratio(failures, train_ratio)
        if train_ratio == 0.3:
            X_val, X_tr, y_val, y_tr = split_by_cutoff(X_seq, y_seq, ts_seq, cutoff_time, put_cutoff_to_train=False)
        else:
            X_tr, X_val, y_tr, y_val = split_by_cutoff(X_seq, y_seq, ts_seq, cutoff_time)
        #print(np.unique(y_val, return_counts=True))
        if model_name == "LSTM":
            model = LSTMModel(input_size=X_tr.shape[2],
                              hidden_size=hidden_size,
                              dropout=dropout,
                              temperature=temperature).to(device)
        elif model_name == "GRU":
            model = GRUModel(input_size=X_tr.shape[2],
                             hidden_size=hidden_size,
                             dropout=dropout,
                             temperature=temperature).to(device)
        elif model_name == "GRU_Att":
            model = GRUWithAttention(input_size=X_tr.shape[2],
                                     hidden_size=hidden_size,
                                     dropout=dropout,
                                     temperature=temperature).to(device)
        elif model_name == "CNV_GRU":
            model = ConvGRU(input_size=X_tr.shape[2],
                            hidden_size=hidden_size,
                            dropout=dropout,
                            temperature=temperature).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

        criterion = lambda logits, targets: focal_loss_ce(logits, targets, alpha=alpha, gamma=gamma)
        train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                                torch.tensor(y_tr, dtype=torch.float32)), batch_size=64, shuffle=False)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                              torch.tensor(y_val, dtype=torch.float32)), batch_size=64, shuffle=False)
        best_f1 = 0
        for epoch in range(1, 100):
            f1, auc, pr, _ = train_and_eval(model, train_loader, val_loader,
                                            optimizer, criterion, device, scheduler)
            best_f1 = max(best_f1, f1)
        fold_scores.append(best_f1)

    score = np.mean(fold_scores)
    trial.report(score, step=1)
    return score

def study_optuna(X_raw, y_raw, timestamps, failures, device, timeout=None, model_name=None):
    # TPE + Media pruner
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner=MedianPruner(
        n_startup_trials=10,     # 초반 10개 trial은 pruning하지 않음
        n_warmup_steps=5,       # 최소 5 step 이후부터 pruning 고려
        interval_steps=1
    )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    objective_with_data = partial(
        objective,
        X_raw=X_raw,
        y_raw=y_raw,
        timestamps=timestamps,
        failures=failures,
        device=device,
        model_name=model_name
    )

    study.optimize(objective_with_data, timeout=timeout)
    print("✅ Best Params:", study.best_params)
    print("✅ Best Composite Score:", study.best_value)
    if len(study.trials) > 1:
        fig1 = vis.plot_optimization_history(study)
        fig1.write_html("tmp/optuna/optuna_optimization_history.html")

        fig2 = vis.plot_param_importances(study)
        fig2.write_html("tmp/optuna/optuna_param_importances.html")

        fig3 = vis.plot_parallel_coordinate(study)
        fig3.write_html("tmp/optuna/optuna_parallel_coordinate.html")
    return study

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
    print(f'Average P-value: {stats_df["p_value"].abs().mean()}')
    return stats_df
