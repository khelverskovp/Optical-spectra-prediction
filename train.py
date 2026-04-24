import os
import sys
import signal
import json
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pandas as pd

from torch.optim.lr_scheduler import ReduceLROnPlateau

# -----------------------
# Optional Weights & Biases
# -----------------------
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from data.dielectricGraphDataset import DielectricGraphDataset
from gotenNet import GotenNet
from config import TRAIN_CFG, GotenNet_Model_CFG

# -----------------------
# Utilities
# -----------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def get_run_dir(exp_num):
    base = ensure_dir("runs")
    run_dir = ensure_dir(os.path.join(base, f"exp_{exp_num}"))
    ensure_dir(os.path.join(run_dir, "checkpoints"))
    ensure_dir(os.path.join(run_dir, "logs"))
    ensure_dir(os.path.join(run_dir, "saved_models"))
    return run_dir

# -----------------------
# Target transform
# -----------------------
class TargetTransform:
    """
    Fitted once on training labels, then frozen.

    Steps (applied in order during encode, reversed during decode):
      1. Offset log:  y → log(y + offset)   if log_transform=True
      2. Standardize: y → (y - mean) / std   if standardize=True

    The log offset is fixed (config: target_log_offset, default 100). This
    safely covers the known dataset minimum of ~-90.5 with comfortable
    headroom and is never computed from data, so there is no risk of val/test
    NaNs from split distribution mismatch.

    mean/std are computed from training labels only (after the optional log
    step), so there is no leakage into val/test.
    """

    def __init__(self, log_transform: bool = False, standardize: bool = False,
                 offset: float = 100.0):
        self.log_transform = log_transform
        self.standardize   = standardize
        self.offset        = offset   # fixed, never fitted from data
        self.mean          = 0.0
        self.std           = 1.0
        self._fitted       = False

    # ---- fitting -------------------------------------------------------
    def fit(self, train_set: Subset) -> "TargetTransform":
        """
        Compute mean/std from training labels only (after optional log step).
        The offset is fixed and never modified here.
        """
        if not self.standardize:
            self._fitted = True
            print(
                f"[TargetTransform] ready — log={self.log_transform} "
                f"offset={self.offset} | standardize=False"
            )
            return self

        all_labels = []
        for sample in train_set:
            lbl = sample["label"].float().numpy()
            all_labels.append(self._apply_log(lbl))
        stacked   = np.concatenate(all_labels)             # [N_train * L]
        self.mean = float(stacked.mean())
        self.std  = float(stacked.std()) or 1.0

        self._fitted = True
        print(
            f"[TargetTransform] fitted — log={self.log_transform} "
            f"offset={self.offset} | standardize=True "
            f"mean={self.mean:.6f} std={self.std:.6f}"
        )
        return self

    # ---- helpers -------------------------------------------------------
    def _apply_log(self, x: np.ndarray) -> np.ndarray:
        return np.log(x + self.offset) if self.log_transform else x

    def _undo_log(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x) - self.offset if self.log_transform else x

    def _apply_log_t(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + self.offset) if self.log_transform else x

    def _undo_log_t(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) - self.offset if self.log_transform else x

    # ---- public API ----------------------------------------------------
    def encode(self, labels: torch.Tensor) -> torch.Tensor:
        """Transform raw labels → model target space. [B, L] in/out."""
        assert self._fitted, "Call fit() before encode()"
        y = self._apply_log_t(labels)
        if self.standardize:
            y = (y - self.mean) / self.std
        return y

    def decode(self, labels: torch.Tensor) -> torch.Tensor:
        """Inverse: model output → original label space. [B, L] in/out."""
        assert self._fitted, "Call fit() before decode()"
        y = labels
        if self.standardize:
            y = y * self.std + self.mean
        return self._undo_log_t(y)

    # ---- serialisation -------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "log_transform": self.log_transform,
            "standardize":   self.standardize,
            "offset":        self.offset,
            "mean":          self.mean,
            "std":           self.std,
            "_fitted":       self._fitted,
        }

    def load_state_dict(self, d: dict) -> None:
        self.log_transform = d["log_transform"]
        self.standardize   = d["standardize"]
        self.offset        = d["offset"]
        self.mean          = d["mean"]
        self.std           = d["std"]
        self._fitted       = d["_fitted"]

# -----------------------
# Loss functions
# -----------------------
def sc_loss(pred, target, eps=1e-8):
    """
    Loss = mean(1 - SC_i), where SC_i = 1 - trapz(|pred-target|) / trapz(|target|)
    Minimising this loss is equivalent to maximising mean SC.
    pred, target: [B, L]
    """
    abs_diff = (pred - target).abs()
    abs_targ = target.abs()

    trapz_diff = ((abs_diff[:, :-1] + abs_diff[:, 1:]) / 2).sum(dim=-1)  # [B]
    trapz_targ = ((abs_targ[:, :-1] + abs_targ[:, 1:]) / 2).sum(dim=-1)  # [B]

    sc = 1.0 - trapz_diff / (trapz_targ + eps)  # [B], higher is better
    return (1.0 - sc).mean()                     # minimise this → maximise SC

def inverse_huber_loss(pred, target, percentile=0.2):
    r = (pred - target).abs()
    delta = torch.quantile(r, percentile).detach()
    loss = torch.where(r <= delta, r, (r**2 + delta**2) / (2 * delta))
    return loss.mean()

_LOSS_REGISTRY = {
    "mae":           lambda p, t, cfg: F.l1_loss(p, t),
    "mse":           lambda p, t, cfg: F.mse_loss(p, t),
    "sc":            lambda p, t, cfg: sc_loss(p, t),
    "inverse_huber": lambda p, t, cfg: inverse_huber_loss(
        p, t, percentile=cfg.get("loss_percentile", 0.2)
    ),
}

def get_loss_fn(cfg):
    key = cfg.get("loss_fn", "mae").lower()
    if key not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss_fn '{key}'. Choose from: {list(_LOSS_REGISTRY)}"
        )
    return lambda pred, target: _LOSS_REGISTRY[key](pred, target, cfg)

# -----------------------
# Formula-based splits
# -----------------------
def formula_split(raw_data, data_df, train_frac=0.8, val_frac=0.1, seed=0):
    """
    Split dataset indices by unique chemical formula so that all polymorphs
    of a given composition end up in the same split. 
    """
    unique_formulas = np.unique(data_df.formula.values)
    np.random.seed(seed)
    np.random.shuffle(unique_formulas)
    n = len(unique_formulas)
    train_form = set(unique_formulas[:int(n * train_frac)])
    val_form   = set(unique_formulas[int(n * train_frac):int(n * (train_frac + val_frac))])
    test_form  = set(unique_formulas[int(n * (train_frac + val_frac)):])

    # build a fast mat_id -> formula lookup
    mat_id_to_formula = dict(zip(data_df["mat_id"].values, data_df["formula"].values))

    train_idx, val_idx, test_idx = [], [], []
    for i, sample in enumerate(raw_data):
        mat_id = sample.mat_id if not isinstance(sample, dict) else sample["mat_id"]
        formula = mat_id_to_formula.get(mat_id)
        if formula is None:
            print(f"Warning: mat_id {mat_id} not found in data_df, skipping.")
            continue
        if formula in train_form:
            train_idx.append(i)
        elif formula in val_form:
            val_idx.append(i)
        elif formula in test_form:
            test_idx.append(i)

    return train_idx, val_idx, test_idx

# -----------------------
# Dataloader helpers
# -----------------------
def collate_graphs(batch):
    """
    Merges a list of per-graph dicts into a single batched dict.
    Node indices in edge_index are offset so they remain globally unique.
    """
    node_offset = 0
    all_coords, all_node_features, all_graph_indices = [], [], []
    all_edge_index, all_edge_vectors, all_edge_lengths, all_labels = [], [], [], []

    for i, graph in enumerate(batch):
        num_nodes = graph["node_coordinates"].size(0)

        all_coords.append(graph["node_coordinates"])
        all_node_features.append(graph["node_features"])
        all_graph_indices.append(torch.full((num_nodes,), i, dtype=torch.long))

        all_edge_index.append(graph["edge_index"] + node_offset)
        all_edge_vectors.append(graph["edge_vectors"])
        all_edge_lengths.append(graph["edge_lengths"])

        all_labels.append(graph["label"].float().unsqueeze(0))  # [1, 4002]

        node_offset += num_nodes

    return {
        "num_nodes":               node_offset,
        "num_graphs":              len(batch),
        "node_coordinates":        torch.cat(all_coords,          dim=0),   # [N_total, 3]
        "node_features":           torch.cat(all_node_features,   dim=0),   # [N_total, 25]
        "edge_index":              torch.cat(all_edge_index,       dim=0),   # [E_total, 2]
        "edge_vectors":            torch.cat(all_edge_vectors,     dim=0),   # [E_total, 3]
        "edge_lengths":            torch.cat(all_edge_lengths,     dim=0),   # [E_total]
        "node_graph_index":        torch.cat(all_graph_indices,    dim=0),   # [N_total]
        "labels":                  torch.cat(all_labels,           dim=0),   # [B, 4002]
    }

def move_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

# -----------------------
# Training / Eval
# -----------------------
def train(model, loader, optimizer, device, epoch, cfg, use_wandb, loss_fn, transform):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad()
        out, eps_imag, eps_real = model(batch)
        # encode raw labels into transform space before computing loss
        targets = transform.encode(batch["labels"])
        loss = loss_fn(out, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["gradient_clipping"])
        optimizer.step()
        total_loss += loss.item()
        if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({"batch_loss": loss.item(), "epoch": epoch})
    return total_loss / len(loader)


def evaluate(model, loader, device, loss_fn, transform):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            out, eps_imag, eps_real = model(batch)
            targets = transform.encode(batch["labels"])
            total_loss += loss_fn(out, targets).item()
    return total_loss / len(loader)

# -----------------------
# Checkpointing
# -----------------------
def rng_state_state_dict():
    return {
        "torch_cpu":  torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy":      np.random.get_state(),
        "python":     random.getstate(),
    }

def load_rng_state(state):
    if state is None:
        return
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("python") is not None:
        random.setstate(state["python"])

def save_checkpoint(run_dir, state, is_best=False, tag="last"):
    ckpt_path = os.path.join(run_dir, "checkpoints", f"{tag}.ckpt")
    torch.save(state, ckpt_path)
    if is_best:
        torch.save(state, os.path.join(run_dir, "checkpoints", "best.ckpt"))
    del state  # explicitly free
    torch.cuda.empty_cache()
    return ckpt_path

def try_load_checkpoint(run_dir, tag="last"):
    ckpt_path = os.path.join(run_dir, "checkpoints", f"{tag}.ckpt")
    if os.path.isfile(ckpt_path):
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return None

# -----------------------
# Main
# -----------------------
def main():
    cfg = TRAIN_CFG

    use_wandb = cfg.get("use_wandb", True) and WANDB_AVAILABLE

    exp_num = cfg["experiment_number"]
    run_dir = get_run_dir(exp_num)

    # -----------------------
    # W&B init (with resume)
    # -----------------------
    if use_wandb:
        wandb_id_path = os.path.join(run_dir, "wandb_run_id.txt")
        wandb_id = None
        if os.path.isfile(wandb_id_path):
            with open(wandb_id_path) as f:
                wandb_id = f.read().strip()

        wandb.init(
            project=cfg.get("wandb_project", "dielectric"),
            entity=cfg.get("wandb_entity", None),
            name=cfg.get("run_name", "gotennet-ipa"),
            resume="allow",
            id=wandb_id,
        )
        if wandb_id is None and wandb.run is not None:
            with open(wandb_id_path, "w") as f:
                f.write(wandb.run.id)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    train_seed = int(cfg.get("train_seed", 0))
    split_seed = int(cfg.get("split_seed", cfg.get("seed", train_seed)))

    set_seed(train_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # Dataset
    # -----------------------
    with open(cfg["dataset_path"], "rb") as f:
        raw_data = pickle.load(f)

    dataset = DielectricGraphDataset(raw_data)
    print(f"Dataset loaded: {len(dataset)} samples")

    data_df = pd.read_pickle(cfg["data_df_path"])
    data_df = data_df.reset_index()
    print(f"Loaded data_df: {len(data_df)} materials")

    # -----------------------
    # Splits (formula-based, reproducible)
    # -----------------------
    splits_path = os.path.join(run_dir, "splits.pkl")
    splits = None

    if os.path.isfile(splits_path):
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        if int(splits.get("split_seed", -1)) != split_seed:
            print(
                f"[INFO] Existing splits used split_seed={splits.get('split_seed')}, "
                f"cfg has split_seed={split_seed}. Regenerating."
            )
            splits = None

    if splits is None:
        train_indices, val_indices, test_indices = formula_split(
            raw_data, data_df,
            train_frac=cfg["train_frac"],
            val_frac=cfg["val_frac"],
            seed=split_seed,
        )
        with open(splits_path, "wb") as f:
            pickle.dump({
                "train":      train_indices,
                "val":        val_indices,
                "test":       test_indices,
                "split_seed": split_seed,
            }, f)
        print(
            f"Created splits: train={len(train_indices)}, "
            f"val={len(val_indices)}, test={len(test_indices)} (seed={split_seed})"
        )
    else:
        train_indices = splits["train"]
        val_indices   = splits["val"]
        test_indices  = splits["test"]
        print(
            f"Loaded existing splits: train={len(train_indices)}, "
            f"val={len(val_indices)}, test={len(test_indices)}"
        )

    train_set = Subset(dataset, train_indices)
    val_set   = Subset(dataset, val_indices)
    test_set  = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"],
        shuffle=True,  collate_fn=collate_graphs, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["batch_size"],
        shuffle=False, collate_fn=collate_graphs, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg["batch_size"],
        shuffle=False, collate_fn=collate_graphs, drop_last=False,
    )

    # -----------------------
    # Model / optimiser / scheduler
    # -----------------------
    model     = GotenNet(GotenNet_Model_CFG).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01)
    )
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15,
        threshold=1e-6, threshold_mode="rel",
        cooldown=2, min_lr=cfg["min_lr"],
    )

    loss_fn = get_loss_fn(cfg)
    print(f"Loss function: {cfg.get('loss_fn', 'mae')}")

    # -----------------------
    # Target transform (fit once on train set)
    # -----------------------
    transform = TargetTransform(
        log_transform=cfg.get("target_log_transform", False),
        standardize=cfg.get("target_standardize", False),
        offset=cfg.get("target_log_offset", 100.0),
    )

    # -----------------------
    # Resume if checkpoint exists
    # -----------------------
    start_epoch       = 1
    best_val_loss     = float("inf")
    patience_counter  = 0
    smoothed_val_loss = None
    train_losses, val_losses = [], []

    ckpt = try_load_checkpoint(run_dir, tag="last")
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        plateau_scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch       = ckpt["epoch"] + 1
        best_val_loss     = ckpt["best_val_loss"]
        patience_counter  = ckpt["patience_counter"]
        smoothed_val_loss = ckpt["smoothed_val_loss"]
        train_losses      = ckpt.get("train_losses", [])
        val_losses        = ckpt.get("val_losses", [])
        load_rng_state(ckpt.get("rng_state"))
        # restore the previously fitted transform (never refit on resume)
        if "transform_state" in ckpt:
            transform.load_state_dict(ckpt["transform_state"])
            print("[TargetTransform] restored from checkpoint")
        else:
            # legacy checkpoint without transform — fit now
            print("[TargetTransform] no transform in checkpoint, fitting now...")
            transform.fit(train_set)
        print(
            f"Resumed from epoch {ckpt['epoch']} "
            f"(best_val_loss={best_val_loss:.6f}, patience={patience_counter})"
        )
    else:
        # fresh run: offset scanned over full dataset so val/test NaNs are impossible;
        # mean/std still computed on train only (no leakage).
        transform.fit(train_set)
        print("Starting fresh training run.")

    # -----------------------
    # Snapshot on interrupt
    # -----------------------
    loss_log_path     = os.path.join(run_dir, "logs", "train_val_losses.pkl")
    test_results_path = os.path.join(run_dir, "logs", "test_results.pkl")
    best_model_path   = os.path.join(run_dir, "saved_models", "best_model_state_dict.pt")

    def _build_state(epoch):
        return {
            "epoch":             epoch,
            "model_state":       model.state_dict(),
            "optimizer_state":   optimizer.state_dict(),
            "scheduler_state":   plateau_scheduler.state_dict(),
            "best_val_loss":     best_val_loss,
            "patience_counter":  patience_counter,
            "smoothed_val_loss": smoothed_val_loss,
            "train_losses":      train_losses,
            "val_losses":        val_losses,
            "rng_state":         rng_state_state_dict(),
            "transform_state":   transform.state_dict(),   # <-- persisted with checkpoint
            "config":            cfg,
        }

    def _snapshot_and_exit(signum, frame):
        state = _build_state(max(start_epoch - 1, 1))
        path = save_checkpoint(run_dir, state, is_best=False, tag="emergency")
        print(f"\nSignal {signum} received — emergency checkpoint saved to {path}. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _snapshot_and_exit)
    signal.signal(signal.SIGINT,  _snapshot_and_exit)

    # -----------------------
    # Training loop
    # -----------------------
    max_epochs   = cfg["max_epochs"]
    alpha        = cfg.get("alpha", 0.9)
    patience     = cfg["patience"]
    spike_factor = cfg.get("spike_factor", 2.0)

    print("=" * 60)
    print(f" {'Resuming' if ckpt else 'Starting'} | device={device} | "
          f"train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    print("=" * 60)

    for epoch in range(start_epoch, max_epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, epoch, cfg, use_wandb, loss_fn, transform)
        val_loss   = evaluate(model, val_loader, device, loss_fn, transform)

        # EMA smoothing (spike-resistant)
        if smoothed_val_loss is None:
            smoothed_val_loss = val_loss
        else:
            if val_loss > spike_factor * smoothed_val_loss:
                print(
                    f"  ⚠️  Val loss spike ignored: {val_loss:.6f} "
                    f"(>{spike_factor}× smoothed {smoothed_val_loss:.6f})"
                )
                effective = smoothed_val_loss
            else:
                effective = val_loss
            smoothed_val_loss = alpha * smoothed_val_loss + (1 - alpha) * effective

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:03d} | train={train_loss:.6f} | "
            f"val={val_loss:.6f} | smoothed={smoothed_val_loss:.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "epoch":             epoch,
                "train_loss":        train_loss,
                "val_loss":          val_loss,
                "smoothed_val_loss": smoothed_val_loss,
                "lr":                optimizer.param_groups[0]["lr"],
            })

        plateau_scheduler.step(smoothed_val_loss)

        state   = _build_state(epoch)
        is_best = smoothed_val_loss < best_val_loss

        if is_best:
            best_val_loss    = smoothed_val_loss
            patience_counter = 0
            save_checkpoint(run_dir, state, is_best=True, tag="last")
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best model saved (val={best_val_loss:.6f})")
        else:
            patience_counter += 1
            save_checkpoint(run_dir, state, is_best=False, tag="last")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    with open(loss_log_path, "wb") as f:
        pickle.dump({"train": train_losses, "val": val_losses}, f)

    # -----------------------
    # Test-set evaluation
    # (decode back to original label space for interpretable MSE/MAE)
    # -----------------------
    best_ckpt = try_load_checkpoint(run_dir, tag="best") or try_load_checkpoint(run_dir, tag="last")
    if best_ckpt is not None:
        model.load_state_dict(best_ckpt["model_state"])
        print("Loaded best checkpoint for test evaluation.")

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = move_to_device(batch, device)
            out, eps_imag, eps_real = model(batch)
            # decode model outputs back to original label space
            all_preds.append(transform.decode(out).cpu())
            all_targets.append(batch["labels"].cpu())

    preds   = torch.cat(all_preds,   dim=0).numpy()   # [N_test, 4002]
    targets = torch.cat(all_targets, dim=0).numpy()   # [N_test, 4002]

    test_mse = float(np.mean((preds - targets) ** 2))
    test_mae = float(np.mean(np.abs(preds - targets)))
    print(f"\nTest MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f}")

    with open(test_results_path, "wb") as f:
        pickle.dump(
            {
                "predictions":    preds,
                "targets":        targets,
                "test_mse":       test_mse,
                "test_mae":       test_mae,
                "transform_state": transform.state_dict(),
            },
            f,
        )

    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({"test_mse": test_mse, "test_mae": test_mae})
        wandb.finish()

    print(f"\nDone. Results saved to {run_dir}")


if __name__ == "__main__":
    main()