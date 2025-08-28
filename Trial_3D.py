# ================= RSNA-MICCAI MGMT • TopoCAM (leakage-free) =================
# - 4→3 adapter in front of pretrained R3D-18
# - Freeze backbone; train adapter + fc head for Pred-CAM targets
# - Grad-CAM targets are predicted classes (no GT on val/test)
# - Persistence on ROI-masked scalar volumes: seg_4d.max(axis=1)
# - BettiCurve fit on TRAIN; transform VAL/TEST
# - Differential Evolution tunes fusion weights on clean VAL features
# - MLP trained on TRAIN features; metrics reported on VAL and TEST
# ============================================================================

import os, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix
)
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve
from scipy.ndimage import zoom
from scipy.optimize import differential_evolution
import pydicom

# ----------------------------- Config -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Data paths
BASE_DIR = "/scratch/09457/bxb210001/BRITIS_2021/train"   # each subject/FLAIR,T1w,T1wCE,T2w
CSV_PATH = "/scratch/09457/bxb210001/BRITIS_2021/train_labels.csv"

# Volumetric settings
MODALITIES   = ["FLAIR", "T1w", "T1wCE", "T2w"]
TARGET_SHAPE = (64, 64, 64)     # (D,H,W)

# Training + optimization
BATCH_TRAIN     = 2             # adjust to GPU memory
HEAD_EPOCHS     = 6             # train adapter+head only
LR_ADAPTER      = 5e-4
LR_HEAD         = 1e-3
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 1.0

# Grad-CAM + DE
CAM_THRESH = 0.6                # threshold on [0,1]
DE_MAXITER = 30
DE_POPSIZE = 15

# TDA
N_BINS    = 50
HOM_DIMS  = [0, 1, 2]

OUT_DIR = "outputs_rsna_topocam_clean"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------- I/O helpers -----------------------------
def load_volume(path):
    # DICOM slices → (D,H,W) float32
    slices = sorted(os.listdir(path))
    vol = [pydicom.dcmread(os.path.join(path, s)).pixel_array for s in slices]
    return np.stack(vol, axis=0).astype(np.float32)

def resize_volume(volume, target_shape=TARGET_SHAPE):
    f = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, f, order=1)

def normalize_per_volume(x):
    # x: (C,D,H,W), scale to [0,1] per subject
    x = x - x.min()
    mx = x.max()
    return x / mx if mx > 0 else x

def process_subject(base_path, brats_id):
    vols = []
    for m in MODALITIES:
        v = load_volume(os.path.join(base_path, brats_id, m))
        v = resize_volume(v)  # (64,64,64)
        vols.append(v)
    arr = np.stack(vols, axis=0)  # (4,64,64,64)
    return normalize_per_volume(arr)

# ----------------------------- Load data -----------------------------
print("Loading subjects...")
df = pd.read_csv(CSV_PATH)
ids = df["BraTS21ID"].astype(str).str.zfill(5)
labels = df["MGMT_value"].values.astype(int)

X, y = [], []
for sid, lab in zip(ids, labels):
    try:
        X.append(process_subject(BASE_DIR, sid))
        y.append(lab)
    except Exception as e:
        print(f"Skipping {sid}: {e}")

X = np.asarray(X, dtype=np.float32)  # (N,4,64,64,64)
y = np.asarray(y, dtype=int)
n_classes = int(np.unique(y).size)
print(f"Loaded: {X.shape}, classes={n_classes}")

# 70/10/20 split (train/val/test)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)
X_va, X_te, y_va, y_te = train_test_split(
    X_tmp, y_tmp, test_size=2/3, stratify=y_tmp, random_state=SEED
)

# ----------------------------- Model (4→3 adapter + R3D-18) -----------------------------
class R3D18WithAdapter(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 1x1x1 adapter: 4ch → 3ch (equal average into each output)
        self.adapter = nn.Conv3d(4, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            w = torch.full((3, 4, 1, 1, 1), 0.25)  # average of 4 input chans
            self.adapter.weight.copy_(w)

        base = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.base = base
        self.fc   = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x):  # x: (B,4,D,H,W)
        x = self.adapter(x)           # (B,3,D,H,W)
        b = self.base
        x = b.stem(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        x = b.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# Instantiate and set requires_grad
model = R3D18WithAdapter(n_classes).to(device)
for p in model.parameters(): p.requires_grad = False
for p in model.adapter.parameters(): p.requires_grad = True
for p in model.fc.parameters():      p.requires_grad = True

# ----------------------------- Train adapter+head -----------------------------
def make_loader(X, y, bs=BATCH_TRAIN, shuffle=True):
    x = torch.tensor(X, dtype=torch.float32)
    t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(x, t), batch_size=bs, shuffle=shuffle, drop_last=False)

opt = torch.optim.Adam(
    [{"params": model.adapter.parameters(), "lr": LR_ADAPTER},
     {"params": model.fc.parameters(),      "lr": LR_HEAD}],
    weight_decay=WEIGHT_DECAY
)
ce = nn.CrossEntropyLoss()

print("Training adapter + head (backbone frozen)...")
dl_tr = make_loader(X_tr, y_tr, BATCH_TRAIN, shuffle=True)
model.train()
for ep in range(HEAD_EPOCHS):
    run_loss = 0.0
    for xb, yb in dl_tr:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = ce(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(list(model.adapter.parameters())+list(model.fc.parameters()), GRAD_CLIP)
        opt.step()
        run_loss += float(loss.item()) * xb.size(0)
    print(f"  epoch {ep+1}/{HEAD_EPOCHS}  CE={run_loss/len(X_tr):.4f}")
model.eval()

@torch.no_grad()
def predict_classes(model, X, bs=2):
    preds = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i+bs], dtype=torch.float32, device=device)
        logits = model(xb)
        preds.append(logits.argmax(dim=1).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)

yhat_tr = predict_classes(model, X_tr)
yhat_va = predict_classes(model, X_va)
yhat_te = predict_classes(model, X_te)

# ----------------------------- Grad-CAM (Pred-CAM) -----------------------------
class MultiLayerFeatureExtractor:
    def __init__(self, model, target_layers):
        self.acts, self.grads, self.hooks = {}, {}, []
        for i, layer in enumerate(target_layers):
            self.hooks.append(layer.register_forward_hook(self._fwd(i)))
            self.hooks.append(layer.register_full_backward_hook(self._bwd(i)))
    def _fwd(self, idx):
        def hook(m, inp, out): self.acts[idx] = out.detach()
        return hook
    def _bwd(self, idx):
        def hook(m, gin, gout): self.grads[idx] = gout[0].detach()
        return hook
    def close(self):
        for h in self.hooks: h.remove()

def compute_weighted_gradcam(model, images, target_classes, layers, weights, threshold=CAM_THRESH):
    """
    Returns:
      masked_cams: (N,D,H,W) float
      masks:       (N,D,H,W) uint8
      segments:    (N,4,D,H,W) float  -> original images masked by 'masks'
    """
    model.eval()
    extractor = MultiLayerFeatureExtractor(model, layers)
    w = torch.tensor(weights, device=device, dtype=torch.float32)

    masked_cams, masks, segments = [], [], []
    for i in range(len(images)):
        vol = torch.tensor(images[i:i+1], dtype=torch.float32, device=device)  # (1,4,D,H,W)
        target = int(target_classes[i])

        model.zero_grad(set_to_none=True)
        logits = model(vol)
        score = logits[:, target].sum()     # NOTE: predicted class; no GT on val/test
        score.backward()

        combined = None
        for idx in range(3):  # 0:layer2, 1:layer3, 2:layer4
            act  = extractor.acts.get(idx)
            grad = extractor.grads.get(idx)
            if act is None or grad is None: continue
            alpha = grad.mean(dim=[0,2,3,4], keepdim=True)     # (1,C,1,1,1)
            cam   = (act * alpha).sum(dim=1, keepdim=True)     # (1,1,d,h,w)
            if combined is None: combined = torch.zeros_like(cam)
            if cam.shape[2:] != combined.shape[2:]:
                cam = F.interpolate(cam, size=combined.shape[2:], mode='trilinear', align_corners=False)
            combined = combined + w[idx] * cam

        cam_np = combined.squeeze().detach().cpu().numpy()      # (d,h,w)
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() + 1e-8)
        d,h,wv = images[i].shape[1:]
        if cam_np.shape != (d,h,wv):
            cam_np = zoom(cam_np, [d/cam_np.shape[0], h/cam_np.shape[1], wv/cam_np.shape[2]], order=1)

        mask = (cam_np > threshold).astype(np.uint8)            # (d,h,w)
        seg4 = images[i] * mask[np.newaxis, ...]                # (4,d,h,w)

        masked_cams.append((cam_np * mask).astype(np.float32))
        masks.append(mask.astype(np.uint8))
        segments.append(seg4.astype(np.float32))

    extractor.close()
    return np.asarray(masked_cams), np.asarray(masks), np.asarray(segments)

# ----------------------------- TDA helpers -----------------------------
cp = CubicalPersistence(homology_dimensions=HOM_DIMS, n_jobs=-1)
bc = BettiCurve(n_bins=N_BINS, n_jobs=-1)

def fit_betti_train_transform_others(vol_tr_scalar, vol_va_scalar=None, vol_te_scalar=None):
    """
    vol_*_scalar: (N,D,H,W) float (after channel-collapse)
    """
    di_tr = cp.fit_transform(vol_tr_scalar)
    bc.fit(di_tr)
    bv_tr = bc.transform(di_tr)
    outs = [bv_tr]
    if vol_va_scalar is not None:
        outs.append(bc.transform(cp.transform(vol_va_scalar)))
    if vol_te_scalar is not None:
        outs.append(bc.transform(cp.transform(vol_te_scalar)))
    return outs if len(outs) > 1 else outs[0]

def betti_to_features(betti, masks_3d):
    """
    Normalize Betti curves by |ROI|, then append |ROI| as last feature.
      betti: (N, n_bins, n_dims) or (N, n_dims, n_bins)  (gtda handles either)
      masks_3d: (N,D,H,W) uint8  -> ROI size per sample
    """
    counts = masks_3d.reshape(len(masks_3d), -1).sum(axis=1).astype(np.float32) + 1e-8  # (N,)
    # Ensure shape (N, n_bins, n_dims)
    if betti.ndim != 3:
        raise ValueError("Unexpected Betti shape")
    # try to detect which axis is bins
    n0, n1, n2 = betti.shape
    # Heuristic: if n1 == N_BINS, assume (N, bins, dims), else swap axes
    if n1 == N_BINS:
        bv = betti
    elif n2 == N_BINS:
        bv = np.transpose(betti, (0, 2, 1))   # (N, bins, dims)
    else:
        # fallback: assume second is bins
        bv = betti
    bv_norm = bv / counts[:, None, None]
    flat = bv_norm.reshape(bv_norm.shape[0], -1)
    return np.concatenate([flat, counts[:, None]], axis=1)

# ----------------------------- Metrics & MLP -----------------------------
def macro_auc_valid(y_true, y_proba, classes):
    aucs = []
    for i, c in enumerate(classes):
        yb = (y_true == c).astype(int)
        pos, neg = yb.sum(), len(yb) - yb.sum()
        if pos > 0 and neg > 0:
            aucs.append(roc_auc_score(yb, y_proba[:, i]))
    return float(np.mean(aucs)) if aucs else np.nan

def sens_spec(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if labels.size == 2:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        tn, fp, fn, tp = cm.ravel()
        return float(tp/(tp+fn+1e-8)), float(tn/(tn+fp+1e-8))
    sens = recall_score(y_true, y_pred, average="macro", zero_division=0)
    specs=[]
    for c in labels:
        ytb = (y_true==c); ypb=(y_pred==c)
        tn, fp, fn, tp = confusion_matrix(ytb, ypb).ravel()
        specs.append(tn/(tn+fp+1e-8) if (tn+fp)>0 else np.nan)
    return float(sens), float(np.nanmean(specs))

def eval_mlp(Xtr, ytr, Xev, yev):
    clf = MLPClassifier(hidden_layer_sizes=(140,100,64), max_iter=500, random_state=SEED)
    clf.fit(Xtr, ytr.ravel())
    ypred = clf.predict(Xev)
    yprob = clf.predict_proba(Xev)
    aucm  = macro_auc_valid(yev, yprob, clf.classes_)
    acc   = accuracy_score(yev, ypred)
    f1m   = f1_score(yev, ypred, average="macro", zero_division=0)
    prec  = precision_score(yev, ypred, average="macro", zero_division=0)
    rec   = recall_score(yev, ypred, average="macro", zero_division=0)
    sens, spec = sens_spec(yev, ypred)
    return {"AUC": aucm, "Accuracy": acc, "F1": f1m, "Precision": prec, "Recall": rec,
            "Sensitivity": sens, "Specificity": spec}

# ----------------------------- Differential Evolution (clean) -----------------------------
layers = [model.base.layer2, model.base.layer3, model.base.layer4]

def de_objective(w):
    w = np.asarray(w, dtype=np.float32)
    if w.sum() <= 0: return 1e6
    w = w / w.sum()

    # Pred-CAM segments (masked originals), + collapse to scalar with max across channels
    _, masks_tr, seg4_tr = compute_weighted_gradcam(model, X_tr, yhat_tr, layers, w, threshold=CAM_THRESH)
    _, masks_va, seg4_va = compute_weighted_gradcam(model, X_va, yhat_va, layers, w, threshold=CAM_THRESH)

    # (N,4,D,H,W) -> (N,D,H,W) via union across modalities
    scal_tr = seg4_tr.max(axis=1).astype(np.float32)
    scal_va = seg4_va.max(axis=1).astype(np.float32)

    # TDA (fit on TRAIN; transform VAL)
    bv_tr, bv_va = fit_betti_train_transform_others(scal_tr, vol_va_scalar=scal_va)

    # Features normalized by |ROI| (use 3D masks for counts)
    feat_tr = betti_to_features(bv_tr, masks_tr)
    feat_va = betti_to_features(bv_va, masks_va)

    # MLP: train on TRAIN, evaluate on VAL (metric to maximize: AUC)
    metrics = eval_mlp(feat_tr, y_tr, feat_va, y_va)
    return -metrics["AUC"]

print("Optimizing fusion weights (DE) on validation features...")
result = differential_evolution(
    de_objective, bounds=[(0,1),(0,1),(0,1)],
    maxiter=DE_MAXITER, popsize=DE_POPSIZE, polish=True, seed=SEED
)
best_w = result.x / (result.x.sum() + 1e-8)
np.save(os.path.join(OUT_DIR, "best_cam_weights.npy"), best_w)
print("Best weights:", best_w)

# ----------------------------- Final features (TRAIN/VAL/TEST) -----------------------------
print("Generating final segments & Betti features...")
# Segments + masks
_, masks_tr, seg4_tr = compute_weighted_gradcam(model, X_tr, yhat_tr, layers, best_w, threshold=CAM_THRESH)
_, masks_va, seg4_va = compute_weighted_gradcam(model, X_va, yhat_va, layers, best_w, threshold=CAM_THRESH)
_, masks_te, seg4_te = compute_weighted_gradcam(model, X_te, yhat_te, layers, best_w, threshold=CAM_THRESH)

# Collapse channels → scalar volumes for TDA
scal_tr = seg4_tr.max(axis=1).astype(np.float32)  # (N,D,H,W)
scal_va = seg4_va.max(axis=1).astype(np.float32)
scal_te = seg4_te.max(axis=1).astype(np.float32)

# Fit cp+bc on TRAIN; transform VAL/TEST
bv_tr, bv_va, bv_te = fit_betti_train_transform_others(scal_tr, vol_va_scalar=scal_va, vol_te_scalar=scal_te)

# Build features (normalize by ROI size; append ROI size)
feat_tr = betti_to_features(bv_tr, masks_tr)
feat_va = betti_to_features(bv_va, masks_va)
feat_te = betti_to_features(bv_te, masks_te)

# Save artifacts
np.savez_compressed(
    os.path.join(OUT_DIR, "masked_segments_and_masks.npz"),
    seg_train=seg4_tr, seg_val=seg4_va, seg_test=seg4_te,
    mask_train=masks_tr, mask_val=masks_va, mask_test=masks_te,
    y_train=y_tr, y_val=y_va, y_test=y_te
)
pd.DataFrame(feat_tr).to_csv(os.path.join(OUT_DIR, "features_train.csv"), index=False)
pd.DataFrame(feat_va).to_csv(os.path.join(OUT_DIR, "features_val.csv"), index=False)
pd.DataFrame(feat_te).to_csv(os.path.join(OUT_DIR, "features_test.csv"), index=False)

# ----------------------------- Evaluation (no train+val concat) -----------------------------
print("Evaluating MLP (train on TRAIN only)...")
val_metrics  = eval_mlp(feat_tr, y_tr, feat_va, y_va)
test_metrics = eval_mlp(feat_tr, y_tr, feat_te, y_te)
print("VAL  metrics:", val_metrics)
print("TEST metrics:", test_metrics)

pd.DataFrame([{
    **{f"w_layer{i+1+1}": best_w[i] for i in range(3)},  # layer2..4
    **{f"VAL_{k}": v for k,v in val_metrics.items()},
    **{f"TEST_{k}": v for k,v in test_metrics.items()},
}]).to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)

print("Done. Artifacts saved to:", os.path.abspath(OUT_DIR))
