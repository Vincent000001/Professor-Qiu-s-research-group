import os
import re
import pickle
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SQL_KEYWORDS = {
    "select", "union", "drop", "insert", "update", "delete", "where", "from",
    "or", "and", "sleep", "benchmark", "information_schema", "concat", "substr",
}
COMMENT_TOKENS = ["--", "#", "/*", "*/"]


@dataclass
class SQLDataBundle:
    x_train: List[str]
    y_train: np.ndarray
    x_val: List[str]
    y_val: np.ndarray
    x_test: List[str]
    y_test: np.ndarray


def simple_clean(text: str) -> str:
    return " ".join(str(text).strip().split())


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    label_candidates = [c for c in df.columns if c.lower() in {"label", "y", "target", "class"}]
    if label_candidates:
        label_col = label_candidates[0]
    else:
        binary_like = []
        for col in df.columns:
            uniq = set(df[col].dropna().astype(str).str.lower().unique().tolist())
            if len(uniq) <= 6 and uniq & {"0", "1", "normal", "sqli", "benign", "malicious"}:
                binary_like.append(col)
        if not binary_like:
            raise ValueError("无法自动识别标签列，请在CSV中使用 label/y/target/class 命名。")
        label_col = binary_like[0]

    text_candidates = [c for c in df.columns if c.lower() in {"text", "payload", "query", "input", "request", "content"}]
    if text_candidates:
        text_col = text_candidates[0]
    else:
        text_col = max((c for c in df.columns if c != label_col), key=lambda x: df[x].astype(str).str.len().mean())
    return text_col, label_col


def normalize_label(v: str) -> int:
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "sqli", "sqli_attack", "malicious", "attack"}:
        return 1
    if s in {"0", "false", "no", "normal", "benign", "clean"}:
        return 0
    try:
        return 1 if float(s) >= 1 else 0
    except Exception:
        return 1 if "sql" in s or "attack" in s else 0


def load_prepared_dataset(base_dir: str = "prepared_sql_dataset") -> SQLDataBundle:
    required = {
        "train": os.path.join(base_dir, "sql_train_augmented.csv"),
        "val": os.path.join(base_dir, "sql_val.csv"),
        "test": os.path.join(base_dir, "sql_test.csv"),
    }
    for k, p in required.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少 {k} 文件: {p}")

    def read_one(path: str):
        df = pd.read_csv(path)
        text_col, label_col = detect_columns(df)
        df = df[[text_col, label_col]].copy()
        df.columns = ["text", "label"]
        df.dropna(subset=["text", "label"], inplace=True)
        df["text"] = df["text"].astype(str).map(simple_clean)
        df["label"] = df["label"].map(normalize_label).astype(int)
        df.drop_duplicates(subset=["text", "label"], inplace=True)
        return df

    train_df, val_df, test_df = [read_one(required[k]) for k in ["train", "val", "test"]]
    return SQLDataBundle(
        x_train=train_df["text"].tolist(), y_train=train_df["label"].to_numpy(),
        x_val=val_df["text"].tolist(), y_val=val_df["label"].to_numpy(),
        x_test=test_df["text"].tolist(), y_test=test_df["label"].to_numpy(),
    )


def global_features(text: str) -> np.ndarray:
    t = text.lower()
    length = max(len(t), 1)
    kw_count = sum(t.count(k) for k in SQL_KEYWORDS)
    special_chars = sum(ch in "'\";()=<>/*!#-+,%" for ch in t)
    quotes = sum(ch in "'\"" for ch in t)
    comments = sum(t.count(tok) for tok in COMMENT_TOKENS)
    digits = sum(ch.isdigit() for ch in t)
    letters = sum(ch.isalpha() for ch in t)
    specials = sum((not ch.isalnum()) for ch in t)
    return np.array([
        length, kw_count, special_chars, quotes, comments,
        digits / length, letters / length, specials / length,
    ], dtype=np.float32)


def local_risk_features(text: str) -> np.ndarray:
    tl = text.lower()
    feats = []
    for i, ch in enumerate(tl):
        near = tl[max(0, i - 5): min(len(tl), i + 6)]
        feats.append([
            1.0 if ch in "'\"" else 0.0,
            1.0 if ch.isdigit() else 0.0,
            1.0 if ch.isalpha() else 0.0,
            1.0 if not ch.isalnum() else 0.0,
            1.0 if any(k in near for k in SQL_KEYWORDS) else 0.0,
            1.0 if any(tok in near for tok in COMMENT_TOKENS) else 0.0,
        ])
    if not feats:
        return np.zeros((1, 6), dtype=np.float32)
    return np.array(feats, dtype=np.float32)


class SQLSequenceDataset(Dataset):
    def __init__(self, texts, labels, max_len=256):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx][: self.max_len]
        token_ids = np.array([ord(c) % 128 for c in text], dtype=np.int64)
        if len(token_ids) == 0:
            token_ids = np.array([0], dtype=np.int64)
        lrf = local_risk_features(text)
        gf = global_features(text)
        return token_ids, lrf, gf, np.int64(self.labels[idx])


def collate_fn(batch):
    toks, lrfs, gfs, ys = zip(*batch)
    max_len = max(len(x) for x in toks)
    bsz = len(batch)
    tok_pad = torch.zeros((bsz, max_len), dtype=torch.long)
    lrf_pad = torch.zeros((bsz, max_len, 6), dtype=torch.float32)
    mask = torch.zeros((bsz, max_len), dtype=torch.float32)
    for i, (t, l) in enumerate(zip(toks, lrfs)):
        tok_pad[i, :len(t)] = torch.from_numpy(t)
        lrf_pad[i, :len(l)] = torch.from_numpy(l)
        mask[i, :len(t)] = 1
    return tok_pad, lrf_pad, torch.tensor(np.stack(gfs), dtype=torch.float32), torch.tensor(ys), mask


class RiskGatedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, local_risk_size=6, global_risk_size=8, alpha=0.5, beta=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = beta
        self.xh = nn.Linear(input_size + hidden_size, hidden_size * 4)
        self.risk_gate = nn.Linear(local_risk_size + global_risk_size, hidden_size)

    def forward(self, x_t, h_prev, c_prev, r_t, g):
        gates = self.xh(torch.cat([x_t, h_prev], dim=-1))
        i_t, f_t, o_t, c_hat = torch.chunk(gates, 4, dim=-1)
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        o_t = torch.sigmoid(o_t)
        c_hat = torch.tanh(c_hat)

        q_t = torch.sigmoid(self.risk_gate(torch.cat([r_t, g], dim=-1)))
        i_prime = torch.clamp(i_t + self.alpha * q_t, 0.0, 1.0)
        f_prime = torch.clamp(f_t + self.beta * q_t, 0.0, 1.0)
        c_t = f_prime * c_prev + i_prime * c_hat
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class RiskGatedLSTMClassifier(nn.Module):
    def __init__(self, vocab_size=128, emb_dim=64, hidden=96, with_local=True, with_global=True, with_attention=True):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.with_local = with_local
        self.with_global = with_global
        self.with_attention = with_attention
        ldim = 6 if with_local else 6
        gdim = 8 if with_global else 8
        self.cell = RiskGatedLSTMCell(emb_dim, hidden, ldim, gdim)
        self.attn = nn.Linear(hidden, 1)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, local_risk, global_risk, mask):
        emb = self.emb(x)
        bsz, seq, _ = emb.shape
        h = torch.zeros((bsz, self.cell.hidden_size), device=emb.device)
        c = torch.zeros_like(h)
        hs = []
        for t in range(seq):
            r_t = local_risk[:, t, :] if self.with_local else torch.zeros_like(local_risk[:, t, :])
            g_t = global_risk if self.with_global else torch.zeros_like(global_risk)
            h, c = self.cell(emb[:, t, :], h, c, r_t, g_t)
            hs.append(h.unsqueeze(1))
        H = torch.cat(hs, dim=1)
        if self.with_attention:
            score = self.attn(H).squeeze(-1)
            score = score.masked_fill(mask == 0, -1e9)
            w = torch.softmax(score, dim=1).unsqueeze(-1)
            rep = (H * w).sum(dim=1)
        else:
            lengths = mask.sum(dim=1).long() - 1
            rep = H[torch.arange(bsz), torch.clamp(lengths, min=0)]
        return self.out(rep).squeeze(-1)


def evaluate_scores(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
    }


def train_torch_model(name, model, train_loader, val_loader, epochs=3, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best = (-1, None)
    for _ in range(epochs):
        model.train()
        for x, lrf, gf, y, m in train_loader:
            x, lrf, gf, y, m = x.to(device), lrf.to(device), gf.to(device), y.float().to(device), m.to(device)
            opt.zero_grad()
            logits = model(x, lrf, gf, m)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
        yv, pv = infer_torch(model, val_loader, device)
        f1 = f1_score(yv, (pv >= 0.5).astype(int), zero_division=0)
        if f1 > best[0]:
            best = (f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model


def infer_torch(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, lrf, gf, y, m in loader:
            logits = model(x.to(device), lrf.to(device), gf.to(device), m.to(device))
            prob = torch.sigmoid(logits).cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(prob.tolist())
    return np.array(ys), np.array(ps)


def main():
    os.makedirs("outputs", exist_ok=True)
    data = load_prepared_dataset("prepared_sql_dataset")

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    Xtr = vec.fit_transform(data.x_train)
    Xte = vec.transform(data.x_test)

    results = []
    probs = {}

    models = {
        "TFIDF_LR": LogisticRegression(max_iter=400),
        "TFIDF_SVM": SVC(probability=True),
        "TFIDF_RF": RandomForestClassifier(n_estimators=200, random_state=SEED),
    }
    for n, m in models.items():
        m.fit(Xtr, data.y_train)
        p = m.predict_proba(Xte)[:, 1]
        probs[n] = p
        met = evaluate_scores(data.y_test, p)
        met["Model"] = n
        results.append(met)

    train_loader = DataLoader(SQLSequenceDataset(data.x_train, data.y_train), batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(SQLSequenceDataset(data.x_val, data.y_val), batch_size=128, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(SQLSequenceDataset(data.x_test, data.y_test), batch_size=128, shuffle=False, collate_fn=collate_fn)

    ablations = {
        "LSTM_base": dict(with_local=False, with_global=False, with_attention=False),
        "LSTM_local": dict(with_local=True, with_global=False, with_attention=False),
        "LSTM_global": dict(with_local=False, with_global=True, with_attention=False),
        "RiskGatedLSTM": dict(with_local=True, with_global=True, with_attention=False),
        "RiskGatedLSTM_Attn": dict(with_local=True, with_global=True, with_attention=True),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for n, cfg in ablations.items():
        model = RiskGatedLSTMClassifier(**cfg)
        model = train_torch_model(n, model, train_loader, val_loader)
        y, p = infer_torch(model, test_loader, device)
        probs[n] = p
        met = evaluate_scores(y, p)
        met["Model"] = n
        results.append(met)

    df = pd.DataFrame(results)[["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]]
    df.to_csv("outputs/sql_injection_detection_results.csv", index=False)

    with open("outputs/model_probabilities.pkl", "wb") as f:
        pickle.dump({"y_test": data.y_test, "probabilities": probs}, f)

    # F1 bar
    plt.figure(figsize=(10, 4))
    plt.bar(df["Model"], df["F1"])
    plt.xticks(rotation=30, ha="right")
    plt.title("F1 Comparison")
    plt.tight_layout()
    plt.savefig("outputs/f1_comparison.png", dpi=160)
    plt.close()

    # multi-metric
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    x = np.arange(len(df))
    w = 0.15
    plt.figure(figsize=(12, 5))
    for i, m in enumerate(metrics):
        plt.bar(x + i * w, df[m], width=w, label=m)
    plt.xticks(x + 2 * w, df["Model"], rotation=30, ha="right")
    plt.legend()
    plt.title("Metrics Comparison")
    plt.tight_layout()
    plt.savefig("outputs/metrics_comparison.png", dpi=160)
    plt.close()

    # ROC & PR (top 4 by F1)
    top_models = df.sort_values("F1", ascending=False).head(4)["Model"].tolist()
    plt.figure(figsize=(6, 5))
    for n in top_models:
        fpr, tpr, _ = roc_curve(data.y_test, probs[n])
        plt.plot(fpr, tpr, label=n)
    plt.plot([0, 1], [0, 1], "k--")
    plt.legend()
    plt.title("ROC Curves")
    plt.tight_layout()
    plt.savefig("outputs/roc_curves.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 5))
    for n in top_models:
        p, r, _ = precision_recall_curve(data.y_test, probs[n])
        plt.plot(r, p, label=n)
    plt.legend()
    plt.title("PR Curves")
    plt.tight_layout()
    plt.savefig("outputs/pr_curves.png", dpi=160)
    plt.close()

    print(df)


if __name__ == "__main__":
    main()
