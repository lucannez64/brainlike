#!/usr/bin/env python3
# enhanced_mc_gpu.py

import cProfile, pstats
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ======================
# Utility / Data Gen
# ======================
def generate_many_simple_tasks_data(
    n_samples=1000, seed=42,
    n_reg_tasks=3, n_binary_clf_tasks=2,
    noise_level_reg=0.05, noise_level_clf_flip_prob=0.02
):
    np.random.seed(seed)
    x0 = np.random.uniform(-2,2,n_samples)
    x1 = np.random.uniform(-2,2,n_samples)
    x2 = np.random.uniform(0,1,n_samples)
    x3 = np.sin(np.pi*x0*0.8) + 0.5*np.cos(np.pi*x1*0.6)
    x4 = x0*x1*np.exp(-0.1*x2**2)
    x5 = np.random.normal(0,0.3,n_samples)
    inputs = np.stack([x0,x1,x2,x3,x4,x5],axis=1).astype(np.float32)

    targets_reg = np.zeros((n_samples,n_reg_tasks),dtype=np.float32)
    if n_reg_tasks>0:
        targets_reg[:,0] = (x0+x1)/2 + 0.5*x2 + np.random.normal(0,noise_level_reg,n_samples)
    if n_reg_tasks>1:
        targets_reg[:,1] = np.tanh(2*x2-1) + 0.3*x3 + np.random.normal(0,noise_level_reg,n_samples)
    if n_reg_tasks>2:
        targets_reg[:,2] = np.cos(np.pi*x3)*(x4/(np.std(x4)+1e-6))*0.5 + np.random.normal(0,noise_level_reg,n_samples)
    # normalize each reg target
    for i in range(n_reg_tasks):
        col=targets_reg[:,i]
        targets_reg[:,i]=(col-col.mean())/(col.std()+1e-6)

    targets_clf = np.zeros((n_samples,n_binary_clf_tasks),dtype=np.float32)
    if n_binary_clf_tasks>0:
        targets_clf[:,0] = (x0>0).astype(np.float32)
    if n_binary_clf_tasks>1:
        targets_clf[:,1] = (x2>0.6).astype(np.float32)
    for i in range(n_binary_clf_tasks):
        flip = np.random.rand(n_samples)<noise_level_clf_flip_prob
        targets_clf[flip,i]=1-targets_clf[flip,i]

    return inputs, targets_reg, targets_clf

# ======================
# Model
# ======================
class MultiTaskComplexLearner(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_regression,
                 n_binary,
                 n_circuits=10,
                 n_internal=5,
                 lr_internal=1e-8,
                 bias_internal_lr=1e-8,
                 internal_decay=0.1,
                 aggregation="mean",
                 activation="leaky_relu",
                 alpha=0.01,
                 use_kwta=False,
                 kwta_k=0,
                 l1_lambda=0.0,
                 device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.n_inputs = n_inputs
        self.n_reg = n_regression
        self.n_bin = n_binary
        self.n_circ = n_circuits
        self.n_int = n_internal
        self.lr_int = lr_internal
        self.b_lr = bias_internal_lr
        self.decay = internal_decay
        self.aggregation = aggregation
        self.activation = activation
        self.alpha = alpha
        self.use_kwta = use_kwta
        self.kwta_k = kwta_k if use_kwta else n_circuits
        self.l1_lambda = l1_lambda

        # Global weights
        self.W1 = nn.Parameter(torch.empty(n_inputs, n_circuits))
        self.b1 = nn.Parameter(torch.zeros(n_circuits))
        if n_regression>0:
            self.W2_reg = nn.Parameter(torch.empty(n_circuits, n_regression))
            self.b2_reg = nn.Parameter(torch.zeros(n_regression))
        if n_binary>0:
            self.W2_clf = nn.Parameter(torch.empty(n_circuits, n_binary))
            self.b2_clf = nn.Parameter(torch.zeros(n_binary))

        # Microcircuit internal weights & biases (not registered as nn.Parameter,
        # updated manually with Hebbian rule)
        self.mc_w = torch.randn(n_circuits, n_internal, device=self.device)*np.sqrt(2.0/n_inputs)
        self.mc_b = torch.randn(n_circuits, n_internal, device=self.device)*0.01

        # Initialize global
        nn.init.kaiming_uniform_(self.W1, a=np.sqrt(5))
        if n_regression>0: nn.init.uniform_(self.W2_reg, -0.1,0.1)
        if n_binary>0:  nn.init.uniform_(self.W2_clf, -0.1,0.1)

    def forward(self, x):
        # x: (batch, n_inputs)
        # 1) Hidden circuit linear
        z = x @ self.W1 + self.b1    # (batch, n_circ)

        # 2) Microcircuit internal pre-activations
        #    Expand z to (batch, n_circ, n_int)
        z_exp = z.unsqueeze(-1)              # (batch, n_circ,1)
        pre = z_exp * self.mc_w + self.mc_b  # (batch, n_circ,n_int)

        # 3) Activation
        if self.activation=="relu":
            act = F.relu(pre)
        else:
            act = F.leaky_relu(pre, negative_slope=self.alpha)

        # 4) Aggregate
        if self.aggregation=="mean":
            h = act.mean(dim=2)   # (batch, n_circ)
        else:
            h,_ = act.max(dim=2)

        # 5) kWTA
        if self.use_kwta:
            # zero all but top-k
            topk, idx = torch.topk(h, self.kwta_k, dim=1)
            mask = torch.zeros_like(h, dtype=torch.bool)
            # idx: (batch,k)
            batch_idx = torch.arange(h.size(0), device=self.device).unsqueeze(1)
            mask[batch_idx, idx] = True
            h = h * mask.float()

        # 6) Heads
        out_reg = h @ self.W2_reg + self.b2_reg if self.n_reg>0 else torch.zeros(x.size(0),0,device=self.device)
        logits_clf = h @ self.W2_clf + self.b2_clf if self.n_bin>0 else torch.zeros(x.size(0),0,device=self.device)
        out_clf = torch.sigmoid(logits_clf) if self.n_bin>0 else out_clf

        return out_reg, logits_clf, h, act

    def hebbian_update(self, z, act):
        # z: (batch, n_circ), act: (batch, n_circ,n_int)
        # pre = z, post = act
        # Δw = lr * mean_over_batch( pre[...,None] * post ) - decay * w
        pre = z.unsqueeze(-1)  # (batch, n_circ,1)
        delta_w = (pre * act).mean(dim=0)   # (n_circ,n_int)
        self.mc_w += self.lr_int * delta_w
        self.mc_w -= self.lr_int * self.decay * self.mc_w

        # Biases: simple activity‐dependent
        delta_b = act.mean(dim=0)  # (n_circ,n_int)
        self.mc_b += self.b_lr * delta_b

    def l1_penalty(self, h):
        return self.l1_lambda * torch.abs(h).sum()

# ======================
# Training / Main
# ======================
def train_and_evaluate(
    model, device,
    train_loader, val_loader,
    n_epochs=1000,
    lr=5e-4,
    loss_weight_reg=1.0,
    loss_weight_clf=1.0,
    clip=1.0,
    verbose=True
):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    history = {"loss":[],"val_loss":[]}

    best_val = float("inf")
    for epoch in range(1, n_epochs+1):
        model.train()
        running_loss = 0.0
        for x,y_reg,y_clf in train_loader:
            x,y_reg,y_clf = x.to(device), y_reg.to(device), y_clf.to(device)
            opt.zero_grad()
            out_reg, logits_clf, h, act = model(x)
            loss = 0.0
            if model.n_reg>0:
                loss_reg = mse(out_reg, y_reg)
                loss += loss_weight_reg * loss_reg
            if model.n_bin>0:
                loss_clf = bce(logits_clf, y_clf)
                loss += loss_weight_clf * loss_clf
            if model.l1_lambda>0:
                loss += model.l1_penalty(h)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            # local Hebbian
            with torch.no_grad():
                z = x @ model.W1 + model.b1
                model.hebbian_update(z, act)

            running_loss += loss.item() * x.size(0)

        train_loss = running_loss/len(train_loader.dataset)

        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x,y_reg,y_clf in val_loader:
                x,y_reg,y_clf = x.to(device), y_reg.to(device), y_clf.to(device)
                out_reg, logits_clf, h, act = model(x)
                loss = 0.0
                if model.n_reg>0:
                    loss += loss_weight_reg * mse(out_reg, y_reg)
                if model.n_bin>0:
                    loss += loss_weight_clf * bce(logits_clf, y_clf)
                if model.l1_lambda>0:
                    loss += model.l1_penalty(h)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if verbose and epoch%50==0:
            print(f"Epoch {epoch}/{n_epochs}: Train {train_loss:.4f}, Val {val_loss:.4f}")
        if val_loss<best_val:
            best_val=val_loss

    return history

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    # Hyperparams
    NUM_SAMPLES = 2000
    TRAIN_RATIO = 0.8
    N_REG = 2
    N_BIN = 1
    N_CIRCUITS = 20
    N_INTERNAL = 4
    BATCH = 32
    EPOCHS = 1000
    LR_GLOBAL = 5e-4
    LR_INT = 1e-8
    BIAS_INT_LR = 1e-8
    DECAY = 0.1

    # Data
    X, Y_reg, Y_clf = generate_many_simple_tasks_data(
        NUM_SAMPLES, seed=2029,
        n_reg_tasks=N_REG,
        n_binary_clf_tasks=N_BIN
    )
    split = int(TRAIN_RATIO*NUM_SAMPLES)
    train_ds = TensorDataset(
        torch.from_numpy(X[:split]),
        torch.from_numpy(Y_reg[:split]),
        torch.from_numpy(Y_clf[:split])
    )
    val_ds = TensorDataset(
        torch.from_numpy(X[split:]),
        torch.from_numpy(Y_reg[split:]),
        torch.from_numpy(Y_clf[split:])
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    # Model
    model = MultiTaskComplexLearner(
        n_inputs=X.shape[1],
        n_regression=N_REG,
        n_binary=N_BIN,
        n_circuits=N_CIRCUITS,
        n_internal=N_INTERNAL,
        lr_internal=LR_INT,
        bias_internal_lr=BIAS_INT_LR,
        internal_decay=DECAY,
        aggregation="max",
        activation="leaky_relu",
        alpha=0.05,
        use_kwta=True,
        kwta_k=8,
        l1_lambda=0.0,
        device=DEVICE
    ).to(DEVICE)

    # Profile training for 1 epoch
    profiler = cProfile.Profile()
    profiler.enable()
    start = time.time()
    history = train_and_evaluate(
        model, DEVICE,
        train_loader, val_loader,
        n_epochs=1, lr=LR_GLOBAL,
        loss_weight_reg=1.0, loss_weight_clf=1.0,
        clip=1.0, verbose=False
    )
    end = time.time()
    profiler.disable()
    print(f"1 epoch time: {end-start:.3f}s")

    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(10)

    # Full training
    history = train_and_evaluate(
        model, DEVICE,
        train_loader, val_loader,
        n_epochs=EPOCHS, lr=LR_GLOBAL,
        loss_weight_reg=1.0, loss_weight_clf=1.0,
        clip=1.0, verbose=True
    )

    # Final evaluation on test split
    X_test = torch.from_numpy(X[split:]).to(DEVICE)
    Y_reg_test = torch.from_numpy(Y_reg[split:]).to(DEVICE)
    Y_clf_test = torch.from_numpy(Y_clf[split:]).to(DEVICE)
    model.eval()
    with torch.no_grad():
        out_reg, logits_clf, h, act = model(X_test)
        preds_clf = (torch.sigmoid(logits_clf)>0.5).float()
    # Compute metrics
    mse1 = F.mse_loss(out_reg[:,0], Y_reg_test[:,0]).item()
    r2_1 = 1 - ((out_reg[:,0]-Y_reg_test[:,0])**2).sum().item() / ((Y_reg_test[:,0]-Y_reg_test[:,0].mean())**2).sum().item()
    mse2 = F.mse_loss(out_reg[:,1], Y_reg_test[:,1]).item()
    r2_2 = 1 - ((out_reg[:,1]-Y_reg_test[:,1])**2).sum().item() / ((Y_reg_test[:,1]-Y_reg_test[:,1].mean())**2).sum().item()
    acc = (preds_clf[:,0]==Y_clf_test[:,0]).float().mean().item()

    print(f"\nTest MSE1 {mse1:.4f} R2_1 {r2_1:.4f}")
    print(f"Test MSE2 {mse2:.4f} R2_2 {r2_2:.4f}")
    print(f"Test Clf Acc {acc:.4f}")

if __name__=="__main__":
    main()
