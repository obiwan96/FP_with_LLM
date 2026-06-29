import re
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx
from pyvis.network import Network
import glob
import imageio.v2 as imageio

# 5G NF graph topology by service connectivity
CORE_NF_ALIASES = ["amf", "ausf", "lmf", "nrf", "smf", "udm", "udr", "upf"]
RAN_NF_ALIASES = ["gnb"]
NODE_NAMES = CORE_NF_ALIASES + RAN_NF_ALIASES
MDAF_FITURES_CANONICAL_DICT = {
    "rrc_state_counts": ['gnb'],
    "ue_failure_counts": ['gnb'],
    "slo_violation": ['gnb', 'amf', 'smf', 'upf'],
    "pdu_session_delay_seconds": ['smf', 'amf'],
    "amf_registration_rate": ['amf'],
    "upf_throughput": ['upf'],
    "smf_session_drop": ['smf']
}

CORE_NF_EDGES = [
    ("amf", "smf"),
    ("amf", "nrf"),
    ("amf", "udm"),
    ("amf", "ausf"),
    ("amf", "lmf"),
    ("smf", "upf"),
    ("nrf", "udm"),
    ("nrf", "udr"),
    ("udm", "udr"),
    ("ausf", "udm"),
]

RAN_EDGES = [
    ("gnb", "amf"),
]

DEFAULT_SERVICE_EDGES = CORE_NF_EDGES + RAN_EDGES
NODE_SUFFIX_PATTERN = re.compile(r"^container_(?P<metric>.+?)(?:_total)?_(?P<node>[A-Za-z0-9-]+)$")


def visual_node_embeddings(model, sample_x, adj, node_names, epoch, device):
    """에포크마다 노드 임베딩이 공간상에서 어떻게 변하는지 관측"""
    model.eval()
    with torch.no_grad():
        # sample_x shape: [batch, time, nodes, features] -> 예: [1, 10, 8, 32]
        x_tensor = torch.tensor(sample_x, dtype=torch.float32, device=device)
        adj = adj.to(device)
        
        # Encoder를 통과한 노드 임베딩 추출 [batch, time, nodes, gnn_hidden]
        embeddings = model.encoder(x_tensor, adj)
        # 시각화를 위해 마지막 타임스텝의 노드 임베딩만 선택 [nodes, gnn_hidden]
        emb_last_step = embeddings[0, -1, :, :].cpu().numpy() 
    '''    
    # 2차원으로 축소
    tsne = TSNE(n_components=2, perplexity=max(2, len(node_names)-1), random_state=42)
    emb_2d = tsne.fit_transform(emb_last_step)
    
    # 그리기
    plt.figure(figsize=(8, 6))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=300, color='skyblue', alpha=0.7)
    
    for i, name in enumerate(node_names):
        plt.text(emb_2d[i, 0], emb_2d[i, 1], name, fontsize=12, ha='center', va='center', weight='bold')
        
    plt.title(f"Node Embeddings Space - Epoch {epoch}")
    plt.grid(True)
    plt.savefig(f"tmp/gnn/embedding_epoch_{epoch}.png") # 이미지로 저장해서 애니메이션(GIF)으로 변환 가능
    plt.close()
    '''
    return emb_last_step

def visualize_all_epochs_at_once(all_embeddings, node_names):
    flat_embeddings = np.vstack(all_embeddings)     
    tsne = TSNE(n_components=2, perplexity=max(2, len(node_names)-1), random_state=42)
    flat_2d = tsne.fit_transform(flat_embeddings)
    emb_2d_split = flat_2d.reshape(len(all_embeddings), len(node_names), 2)    
    for epoch_idx, emb_2d in enumerate(emb_2d_split):
        plt.figure(figsize=(8, 6))
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=300, color='skyblue', alpha=0.7)
        
        for i, name in enumerate(node_names):
            plt.text(emb_2d[i, 0], emb_2d[i, 1], name, fontsize=12, ha='center', va='center', weight='bold')
            
        plt.title(f"Node Embeddings Space - Epoch {epoch_idx*20}")
        plt.grid(True)
        plt.savefig(f"tmp/gnn/embedding_epoch_{epoch_idx*20}.png") # 이미지로 저장해서 애니메이션(GIF)으로 변환 가능
        plt.close()

def visualize_topology_dynamics(adj_matrix, node_scores, node_names, filename="graph_status.html"):
    """
    pyvis를 이용한 html 구현
    adj_matrix: build_nf_adjacency의 결과 (Tensor 혹은 numpy)
    node_scores: explain_node_contributions()의 딕셔너리 결과 {'amf': 0.85, 'smf': 0.12, ...}
    """
    G = nx.Graph()
    
    # 1. 노드 추가 (모델이 예측한 중요도에 따라 크기(size)를 다르게 설정)
    max_score = max(node_scores.values()) if node_scores.values() else 1.0
    for node in node_names:
        score = node_scores.get(node, 0.0)
        # 중요할수록 크고 붉은색에 가깝게 설정
        size = 15 + (score / (max_score + 1e-6)) * 40 
        color = f"rgba(255, {int(255 * (1 - score/max_score))}, 0, 0.8)"
        
        G.add_node(node, size=size, color=color, title=f"Saliency Score: {score:.4f}")
    
    # 2. 엣지 추가 (Adjacency Matrix 기반)
    adj_np = adj_matrix.cpu().numpy() if hasattr(adj_matrix, "cpu") else adj_matrix
    for i, src in enumerate(node_names):
        for j, dst in enumerate(node_names):
            if i != j and adj_np[i, j] > 0:
                # 실제 연결이 존재하는 5G 인터페이스 표현
                G.add_edge(src, dst, width=2, color="#CCCCCC")
                
    # 3. PyVis를 활용해 웹 브라우저에서 볼 수 있는 실물 그래프 생성
    net = Network(notebook=False, heading="5G NF Graph Learning Dynamics", directed=False)
    net.from_nx(G)
    net.toggle_physics(True) # 노드들이 살아 움직이는 물리 효과 활성화
    net.save_graph(filename)

def canonical_nf_name(node_suffix: str) -> Optional[str]:
    lower = node_suffix.lower()
    for alias in CORE_NF_ALIASES + RAN_NF_ALIASES:
        if lower.endswith(alias) or alias in lower:
            return alias
    return None

def make_gif_from_embeddings(img_dir="tmp/gnn", output_gif="tmp/gnn/embedding_dynamics.gif"):
    # 파일 정렬 (embedding_epoch_0.png, 20.png ... 순서대로 긁어옴)
    search_path = f"{img_dir}/embedding_epoch_*.png"
    files = sorted(glob.glob(search_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not files:
        print("만들 이미지 파일이 없습니다.")
        return
        
    images = [imageio.imread(f) for f in files]
    # fps=2 (초당 2장씩 재생)
    imageio.mimsave(output_gif, images, fps=2)
    print(f"성공! {output_gif} 파일이 생성되었습니다.")


def detect_node_suffixes(columns: List[str], single_domain: Optional[str] = None) -> List[str]:
    canonical_to_suffix = {}
    for col in columns:
        match = NODE_SUFFIX_PATTERN.match(col)
        if not match:
            continue
        node_suffix = match.group("node")
        canonical = canonical_nf_name(node_suffix)
        if canonical is None:
            continue
        if single_domain == "core" and canonical not in CORE_NF_ALIASES:
            continue
        if single_domain == "ran" and canonical not in RAN_NF_ALIASES:
            continue
        if canonical not in canonical_to_suffix:
            canonical_to_suffix[canonical] = node_suffix
    # preserve core order first, then RAN
    ordered = []
    for alias in CORE_NF_ALIASES + RAN_NF_ALIASES:
        if alias in canonical_to_suffix:
            ordered.append(canonical_to_suffix[alias])
    return ordered


def build_nf_adjacency(node_suffixes: List[str]) -> torch.Tensor:
    #NF 간 서비스 연결 기반 adjacency matrix 생성
    if not node_suffixes:
        raise ValueError("No node suffixes were detected for graph construction.")

    N = len(node_suffixes)
    adj = np.zeros((N, N), dtype=np.float32)
    canonical_nodes = [canonical_nf_name(node) for node in node_suffixes]

    for i, source in enumerate(canonical_nodes):
        for j, target in enumerate(canonical_nodes):
            if source is None or target is None:
                continue
            if source == target:
                continue
            if (source, target) in DEFAULT_SERVICE_EDGES or (target, source) in DEFAULT_SERVICE_EDGES:
                adj[i, j] = 1.0

    np.fill_diagonal(adj, 1.0)
    degree = np.sum(adj, axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-6)))
    adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt
    return torch.tensor(adj_norm, dtype=torch.float32)


def build_nf_node_features(feats_df: pd.DataFrame, single_domain: Optional[str] = None) -> Tuple[np.ndarray, List[str], List[str], torch.Tensor]:
    # pod-level NF feature를 [time, nodes, features] 텐서로 변환
    node_suffixes = detect_node_suffixes(list(feats_df.columns), single_domain)
    if not node_suffixes:
        raise ValueError("Could not detect any NF node-specific columns. Use pod-level granularity and a supported domain.")

    groups = {node: {} for node in node_suffixes}
    metric_names = set()
    for col in feats_df.columns:
        match = NODE_SUFFIX_PATTERN.match(col)
        if not match:
            if col in MDAF_FITURES_CANONICAL_DICT:
                nf_list = MDAF_FITURES_CANONICAL_DICT[col]
                print(col, nf_list)
                for nf in nf_list:
                    if single_domain is None or (single_domain == "core" and nf in CORE_NF_ALIASES) or (single_domain == "ran" and nf in RAN_NF_ALIASES):
                        for node_suffix in node_suffixes:
                            if canonical_nf_name(node_suffix) == nf:
                                groups[node_suffix][col] = col
                                metric_names.add(col)
            continue
        node_suffix = match.group("node")
        metric = match.group("metric")
        if node_suffix not in groups:
            continue
        groups[node_suffix][metric] = col
        metric_names.add(metric)

    metric_names = sorted(metric_names)
    X = np.zeros((len(feats_df), len(node_suffixes), len(metric_names)), dtype=np.float32)
    for node_idx, node_suffix in enumerate(node_suffixes):
        for metric_idx, metric in enumerate(metric_names):
            col = groups[node_suffix].get(metric)
            if col is not None:
                X[:, node_idx, metric_idx] = feats_df[col].fillna(0.0).astype(float).values

    adjacency = build_nf_adjacency(node_suffixes)
    return X, node_suffixes, metric_names, adjacency


def make_soft_graph_dataset(X: np.ndarray, y: np.ndarray, timestamps: np.ndarray,
                            window: int, horizon: int,
                            smooth_window: int = 3,
                            mode: str = "linear") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #GNN용 soft label 시퀀스 생성
    xs, ys, ts_valid = [], [], []
    N = len(X)
    for i in range(N - window - horizon + 1):
        x_window = X[i : i + window]
        if horizon > 0 and np.any(y[i : i + window] > 0):
            continue

        future_window = y[i + window - 1 : i + window + horizon - 1] if horizon > 0 else np.array([])
        label = 1 if np.any(future_window > 0) else 0

        if not label:
            future_y = y[i + window + horizon : i + window + horizon + smooth_window]
            if np.any(future_y > 0):
                dist = np.argmax(future_y > 0)
                if mode == "linear":
                    label = max(0, 1 - dist / smooth_window)
                elif mode == "exp":
                    label = np.exp(-dist / smooth_window)

        xs.append(x_window)
        ys.append(label)
        ts_valid.append(timestamps[i + window + horizon - 1])

    return np.array(xs), np.array(ys), np.array(ts_valid)


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = x @ self.weight
        out = adj.matmul(support)
        if self.bias is not None:
            out = out + self.bias
        return out


class GNNEncoder(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GraphConvolution(in_features, hidden_features)
        self.conv2 = GraphConvolution(hidden_features, hidden_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, adj)
        return F.relu(x)


class GNNGRUModel(nn.Module):
    def __init__(self, node_feature_dim: int, gnn_hidden: int = 128, gru_hidden: int = 128,
                 num_layers: int = 1, dropout: float = 0.3, temperature: float = 0.6):
        super().__init__()
        self.requires_graph = True
        self.encoder = GNNEncoder(node_feature_dim, gnn_hidden, dropout=dropout)
        self.attn = nn.Linear(gnn_hidden, 1)
        self.gru = nn.GRU(gnn_hidden, gru_hidden, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(gru_hidden, 1)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, nodes, features]
        batch_size, seq_len, _, _ = x.shape
        x = self.encoder(x, adj)
        attn_logits = self.attn(x).squeeze(-1)
        node_weights = torch.softmax(attn_logits, dim=-1)
        graph_repr = (x * node_weights.unsqueeze(-1)).sum(dim=2)
        out, _ = self.gru(graph_repr)
        out = out.mean(dim=1)
        return self.fc(out).squeeze(-1) / self.temperature

    def node_attention(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_encoded = self.encoder(x, adj)
            attn_logits = self.attn(x_encoded).squeeze(-1)
            return torch.softmax(attn_logits, dim=-1)

def explain_node_contributions(model: GNNGRUModel, x: np.ndarray, adj: torch.Tensor,
                               node_names: List[str], device: torch.device) -> dict:
    model.train()
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)
    adj = adj.to(device)
    out = model(x_tensor, adj)
    if out.dim() > 0:
        out = out.sum()
    out.backward()
    saliency = x_tensor.grad.abs().mean(dim=(0, 1, 3)).cpu().numpy()
    return {node: float(score) for node, score in zip(node_names, saliency)}
