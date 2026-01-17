"""
ICSR 2026 - PexFormer Ablation Study
============================================
Variants:
A. PexFormer (Ours) (SOTA) - Full: Patch + Random + SPA
B. w/ MI Sorting           - Patch + MI + SPA
C. w/o SPA                  - Patch + Random + Standard Attn
"""

import pandas as pd
import numpy as np
import time
import os
import warnings
import math
import gc
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    balanced_accuracy_score, matthews_corrcoef
)
from sklearn.feature_selection import mutual_info_classif

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')

# Config
DATA_DIR = '2_07-11-2025'
TARGET_COL = 'Location'
WINDOW_SIZE = 10
BATCH_SIZE = 256
EPOCHS = 30
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 🧱 Model Components
# ==========================================
def attenuated_kaiming_uniform_(tensor, a=math.sqrt(5), scale=1., mode='fan_in', nonlinearity='leaky_relu'):
    fan = nn_init._calculate_correct_fan(tensor, mode)
    gain = nn_init.calculate_gain(nonlinearity, a)
    std = gain * scale / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad(): return tensor.uniform_(-bound, bound)

# --- 1. Tokenizers ---
class PatchVectorTokenizer(nn.Module):
    def __init__(self, window_size, d_token):
        super().__init__()
        self.projector = nn.Linear(window_size, d_token)
        attenuated_kaiming_uniform_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)
    def forward(self, x): return self.projector(x)

class ScalarTokenizer(nn.Module):
    def __init__(self, d_numerical, d_token, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(d_numerical, d_token))
        self.weight2 = nn.Parameter(torch.Tensor(d_numerical, d_token))
        self.bias = nn.Parameter(torch.Tensor(d_numerical, d_token)) if bias else None
        self.bias2 = nn.Parameter(torch.Tensor(d_numerical, d_token)) if bias else None
        attenuated_kaiming_uniform_(self.weight); attenuated_kaiming_uniform_(self.weight2)
        if bias: nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5)); nn_init.kaiming_uniform_(self.bias2, a=math.sqrt(5))
    def forward(self, x_num):
        x1 = self.weight[None] * x_num[:, :, None] + (self.bias[None] if self.bias is not None else 0)
        x2 = self.weight2[None] * x_num[:, :, None] + (self.bias2[None] if self.bias2 is not None else 0)
        return x1 * torch.tanh(x2)

# --- 2. Attention Mechanisms ---
class SemiPermeableAttention(nn.Module):
    def __init__(self, d, n_heads, dropout, init_scale=0.01):
        super().__init__()
        self.n_heads = n_heads
        self.W_q = nn.Linear(d, d); self.W_k = nn.Linear(d, d); self.W_v = nn.Linear(d, d); self.W_out = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout) if dropout else None
        for m in [self.W_q, self.W_k, self.W_v]: attenuated_kaiming_uniform_(m.weight, scale=init_scale); nn_init.zeros_(m.bias)
        attenuated_kaiming_uniform_(self.W_out.weight); nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        bs, n, d = x.shape
        return x.reshape(bs, n, self.n_heads, d//self.n_heads).transpose(1, 2).reshape(bs*self.n_heads, n, -1)

    def forward(self, x_q, x_kv):
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        q, k, v = self._reshape(q), self._reshape(k), self._reshape(v)
        scores = q @ k.transpose(1, 2) / math.sqrt(k.shape[-1])
        # Causal Mask
        seq_ids = torch.arange(q.shape[1], device=q.device)
        mask = (seq_ids[None, None, :] <= seq_ids[None, :, None])
        mask = (1.0 - mask.float()) * -1e4
        scores = scores + mask
        probs = F.softmax(scores, dim=-1)
        if self.dropout: probs = self.dropout(probs)
        out = (probs @ v).reshape(x_q.shape[0], self.n_heads, x_q.shape[1], -1).transpose(1, 2).reshape(x_q.shape[0], x_q.shape[1], -1)
        return self.W_out(out)

class StandardAttention(nn.Module):
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
    def forward(self, x_q, x_kv):
        out, _ = self.mha(x_q, x_kv, x_kv)
        return out

class GLU(nn.Module):
    def __init__(self, d, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(d, d, bias=bias)
        self.linear2 = nn.Linear(d, d, bias=bias)
    def forward(self, x):
        return self.linear1(x) * torch.tanh(self.linear2(x))

# --- 3. Dynamic Model ---
class PexFormer(nn.Module):
    def __init__(self, input_dim, num_classes, variant_config):
        super().__init__()
        self.cfg = variant_config
        dim = 256; depth = 6; heads = 16; dropout = 0.1 # Standardized Config (Same as Baseline)
        
        # Tokenizer Switch
        if self.cfg['tokenizer'] == 'patch':
            self.tokenizer = PatchVectorTokenizer(window_size=input_dim, d_token=dim)
        else: # Scalar
            self.tokenizer = ScalarTokenizer(d_numerical=input_dim, d_token=dim, bias=True)
            
        # Backbone
        self.layers = nn.ModuleList()
        # Layer: SPA -> GLU
        for _ in range(depth):
            if self.cfg['attn'] == 'spa':
                attn = SemiPermeableAttention(dim, heads, dropout)
            else:
                attn = StandardAttention(dim, heads, dropout)
                
            self.layers.append(nn.ModuleDict({
                'attn': attn,
                'norm1': nn.LayerNorm(dim), 
                'glu': GLU(dim), 
                'norm2': nn.LayerNorm(dim)
            }))
        self.last_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x is either (B, S, W) for patch or (B, F) for scalar
        # Tokenizer handles the diff
        x = self.tokenizer(x)
        
        for layer in self.layers:
            # SPA Block
            residual = x
            x_norm = layer['norm1'](x)
            x_attn = layer['attn'](x_norm, x_norm)
            x = residual + x_attn
            
            # GLU Block
            residual = x
            x_norm = layer['norm2'](x)
            x_glu = layer['glu'](x_norm)
            x = residual + x_glu
            
        x = self.last_norm(x)
        x = x.mean(dim=1) 
        return self.head(x)

# ==========================================
# 🛠 Data Pipeline
# ==========================================
def run_ablation():
    print("🔬 Starting PexFormer Ablation Study (SOTA Aligned)...")
    
    # 1. Load Raw
    all_dfs = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith('.xlsx'): all_dfs.append(pd.read_excel(os.path.join(DATA_DIR, f)))
    df = pd.concat(all_dfs, ignore_index=True)
    val_cols = [c for c in df.columns if 'value' in str(c)]
    tgt = [c for c in df.columns if str(c).lower() == TARGET_COL.lower()][0]
    df = df[val_cols+[tgt]].dropna()
    
    def clean(val):
        s=str(val).lower(); 
        for k,v in [('transition','Transition'),('bath','Bathroom'),('bed','Bedroom'),('kitchen','Kitchen'),('sofa','Living/Dining'),('dining','Living/Dining'),('office','Office'),('door','Door')]:
            if k in s: return v
        return 'Other'
    df[tgt] = df[tgt].apply(clean)
    le = LabelEncoder(); y_raw = le.fit_transform(df[tgt].astype(str))
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_raw = scaler.fit_transform(df[val_cols].values) #(TotalTime, Sensors)
    
    # ⚠️ CRITICAL ALIGNMENT ⚠️
    # Capture RNG State after Data Gen to match Baseline's flow exactly.
    # Baseline flow: Seed 42 -> QT -> Shuffle -> Split.
    # We must replicate "State after QT" for EVERY variant.
    rng_state_numpy = np.random.get_state()
    rng_state_torch = torch.get_rng_state()
    
    results = []
    
    # Same Variants (Redesigned for New SOTA)
    variants = [
        {'id': 'A', 'name': 'PexFormer (SOTA)', 'tokenizer': 'patch', 'attn': 'spa', 'sort': 'random', 'flatten': False},
        {'id': 'B', 'name': 'w/ MI Sorting',            'tokenizer': 'patch', 'attn': 'spa', 'sort': 'mi',     'flatten': False},
        {'id': 'C', 'name': 'w/o SPA',                  'tokenizer': 'patch', 'attn': 'std', 'sort': 'random', 'flatten': False},
    ]
    
    EPOCHS_ABLATION = 50 # Match Baseline SOTA
    
    for v in variants:
        # Restore the "Post-QT" Random State to match Baseline
        np.random.set_state(rng_state_numpy)
        torch.set_rng_state(rng_state_torch)
        
        print(f"\n👉 Running Variant {v['id']}: {v['name']}")
        
        # A. Data Prep
        if v['flatten']:
            Xs, ys = [], []
            for i in range(len(X_raw)-WINDOW_SIZE):
                Xs.append(X_raw[i:i+WINDOW_SIZE].flatten())
                ys.append(y_raw[i+WINDOW_SIZE])
            X_input = np.array(Xs)
            input_dim_arg = X_input.shape[1] 
        else:
            Xs, ys = [], []
            for i in range(len(X_raw)-WINDOW_SIZE):
                Xs.append(X_raw[i:i+WINDOW_SIZE].T)
                ys.append(y_raw[i+WINDOW_SIZE])
            X_input = np.array(Xs) 
            input_dim_arg = WINDOW_SIZE
            
        y_seq = np.array(ys)
        
        # B. Sorting
        if v['sort'] == 'mi':
            print("   -> Sorting Features by MI (Variant B)...")
            idx = np.random.choice(len(X_input), min(10000, len(X_input)), replace=False)
            if v['flatten']:
                mi = mutual_info_classif(X_input[idx], y_seq[idx], random_state=42)
                sort_idx = np.argsort(mi)[::-1]
                X_input = X_input[:, sort_idx]
            else:
                N, S, W = X_input.shape
                X_flat_tmp = X_input.reshape(N, -1)
                mi_flat = mutual_info_classif(X_flat_tmp[idx], y_seq[idx], random_state=42)
                mi_sens = mi_flat.reshape(S, W).mean(1)
                sort_idx = np.argsort(mi_sens)[::-1]
                X_input = X_input[:, sort_idx, :]
                
        elif v['sort'] == 'random':
             print("   -> 🎲 Random feature order (Variant A - SOTA)...")
             # Strict Replication of run_baseline_excelformer.py logic
             # num_sensors = X_input.shape[1]
             # perm = np.arange(num_sensors); np.random.shuffle(perm)
             # But here we need to handle Flatten vs Patch
             if v['flatten']:
                 perm = np.arange(X_input.shape[1])
                 np.random.shuffle(perm)
                 X_input = X_input[:, perm]
             else:
                 perm = np.arange(X_input.shape[1])
                 np.random.shuffle(perm)
                 X_input = X_input[:, perm, :]

        # C. Split (Shuffle first!)
        idx = np.arange(len(X_input)); np.random.shuffle(idx)
        X_shuf, y_shuf = X_input[idx], y_seq[idx]
        split = int(len(X_shuf)*0.8)
        X_train, X_test = X_shuf[:split], X_shuf[split:]
        y_train, y_test = y_shuf[:split], y_shuf[split:]
        
        # D. Train
        batch_size = 64 if v['flatten'] else BATCH_SIZE
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=batch_size*2, shuffle=False)
        
        model = PexFormer(input_dim_arg, len(le.classes_), v).to(DEVICE)
        
        # SOTA Hyperparameters (Matches Baseline)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_ABLATION)
        scaler = GradScaler() # Use AMP
        crit = nn.CrossEntropyLoss()
        
        t0 = time.time()
        for ep in range(EPOCHS_ABLATION):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                
                # AMP Context
                with autocast():
                    out = model(bx)
                    loss = crit(out, by)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            scheduler.step() # Step Scheduler
        
        train_time = time.time() - t0
        
        # E. Eval
        model.eval()
        preds = []
        with torch.no_grad():
            for bx, by in test_loader:
                out = model(bx.to(DEVICE))
                preds.extend(out.argmax(1).cpu().numpy())
                
        acc = accuracy_score(y_test, preds)
        
        results.append({
            'Variant': v['name'],
            "Accuracy": acc,
            "F1-Weighted": f1_score(y_test, preds, average='weighted'),
            "F1-Macro": f1_score(y_test, preds, average='macro'),
            "Precision-W": precision_score(y_test, preds, average='weighted', zero_division=0),
            "Recall-W": recall_score(y_test, preds, average='weighted', zero_division=0),
            "Recall-Macro": recall_score(y_test, preds, average='macro', zero_division=0),
            "Balanced Acc": balanced_accuracy_score(y_test, preds),
            "MCC": matthews_corrcoef(y_test, preds),
            "Time": train_time
        })
        
        print(f"   -> Result: Acc {acc:.4f} | F1 {results[-1]['F1-Macro']:.4f} | Time {train_time:.1f}s")
        
        del model, train_loader, test_loader; torch.cuda.empty_cache(); gc.collect()

    print("\n" + "="*60)
    print("🧪 ABLATION STUDY RESULTS")
    print("="*60)
    df_res = pd.DataFrame(results)
    print(df_res.to_markdown(index=False, floatfmt=".4f"))
    
    # 💾 Save Results (Overwrite)
    csv_file = "ablation_results.csv"
    df_res["Timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    # Reorder cols to put timestamp first
    cols = ["Timestamp"] + [c for c in df_res.columns if c != "Timestamp"]
    df_res = df_res[cols]
    
    df_res.to_csv(csv_file, index=False)
    print(f"\n[Saved] Results overwritten to {csv_file}")
    
if __name__ == '__main__':
    run_ablation()
