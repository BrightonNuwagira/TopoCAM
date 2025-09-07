import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.models.video import r3d_18
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.optimize import differential_evolution
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm
import gc

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/work/09457/bxb210001/ls6/3D_UPDATED_project/synapsemnist3d_64.npz"
SAVE_DIR = "synapse3d_topocam"
os.makedirs(SAVE_DIR, exist_ok=True)
BATCH_SIZE = 32
EPOCHS = 50
THRESHOLD = 0.6
HOM_DIMS = [0, 1, 2]
N_BINS = 50


# === LOAD DATA ===
data = np.load(DATA_PATH)
X_train, y_train = data["train_images"], data["train_labels"]
X_val, y_val = data["val_images"], data["val_labels"]
X_test, y_test = data["test_images"], data["test_labels"]
y_train = y_train.astype(int)
y_val = y_val.astype(int)
y_test = y_test.astype(int)
n_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))






# === DATASET ===
class VolumeDataset(Dataset):
    def __init__(self, volumes, labels):
        self.volumes = volumes  # shape: [N, D, H, W]
        self.labels = labels

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        vol = self.volumes[idx]  # [D, H, W]
        vol_tensor = torch.tensor(vol, dtype=torch.float32)  # [D, H, W]

        # Reshape to match R3D-18 input: [3, T, H, W]
        vol_tensor = vol_tensor.unsqueeze(0).permute(0, 2, 3, 1)  # [1, T, H, W]
        vol_tensor = vol_tensor.repeat(3, 1, 1, 1)                # [3, T, H, W]

        return vol_tensor, self.labels[idx]








train_loader = DataLoader(VolumeDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(VolumeDataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(VolumeDataset(X_test, y_test), batch_size=BATCH_SIZE)

# === MODEL ===
def build_model(n_classes):
    model = r3d_18(weights="R3D_18_Weights.KINETICS400_V1")
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(DEVICE)

# === TRAIN CNN ===
def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).view(-1).long()
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE).view(-1).long()
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total * 100
        print(f"Epoch {epoch+1}: Val Acc = {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))

# === GRADCAM ===
def get_gradcams(model, input_tensor, class_idx):
    target_layers = [model.layer2, model.layer3, model.layer4]
    cams = []
    for layer in target_layers:
        cam = GradCAM(model=model, target_layers=[layer])
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)])
        cams.append(grayscale_cam[0] / (np.max(grayscale_cam[0]) + 1e-8))
    return cams

def fuse_cams(cams, weights):
    return sum(w * cam for w, cam in zip(weights, cams))

def segment_volume(volume, fused_cam, tau=THRESHOLD):
    mask = (fused_cam >= tau).astype(np.uint8)
    return volume * mask, mask

# === TOPOLOGY ===
cp = CubicalPersistence(homology_dimensions=HOM_DIMS, n_jobs=-1)
bc = BettiCurve(n_bins=N_BINS, n_jobs=-1)

def compute_betti_features(gray_volume, mask):
    nonzero = np.count_nonzero(mask)
    if nonzero == 0:
        raise ValueError("Empty mask")
    di = cp.fit_transform([gray_volume])
    bv = bc.fit_transform(di)[0]
    norm_bv = bv / nonzero
    return np.append(norm_bv, nonzero)

# === MLP CLASSIFIER ===
class TopologyMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# === OPTIMIZATION ===
def optimize_cam_weights(cams, val_images, val_labels, model):
    def objective(weights):
        weights = np.array(weights)
        weights /= weights.sum() + 1e-8
        val_feats = []

        for img, label in zip(val_images, val_labels):
            try:
                img_tensor = torch.tensor(img, dtype=torch.float32)           # [D, H, W]
                img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)      # [1, T, H, W]
                img_tensor = img_tensor.repeat(3, 1, 1, 1)                     # [3, T, H, W]
                img_tensor = img_tensor.unsqueeze(0).to(DEVICE)               # [1, 3, T, H, W]

                with torch.no_grad():
                    out = model(img_tensor)
                    class_idx = out.argmax(dim=1).item()

                val_cams = get_gradcams(model, img_tensor, class_idx)
                fused = fuse_cams(val_cams, weights)
                segmented, mask = segment_volume(img, fused)
                gray = segmented.squeeze().astype(np.float32)
                feat = compute_betti_features(gray, mask)
                val_feats.append(np.append(feat, label))

            except Exception as e:
                print(f"⚠️ Skipping validation sample: {type(e).__name__} - {e}")
                continue

        if len(val_feats) < 5:
            return 1e6

        val_feats = np.array(val_feats)
        X_val_topo = val_feats[:, :-1]
        y_val_topo = val_feats[:, -1].astype(int)

        clf = TopologyMLP(X_val_topo.shape[1], len(np.unique(y_val_topo))).to(DEVICE)
        opt = torch.optim.Adam(clf.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()

        clf.train()
        for _ in range(5):
            xb = torch.tensor(X_val_topo, dtype=torch.float32).to(DEVICE)
            yb = torch.tensor(y_val_topo, dtype=torch.long).to(DEVICE)
            opt.zero_grad()
            out = clf(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()

        clf.eval()
        with torch.no_grad():
            probs = F.softmax(clf(xb), dim=1)[:, 1].cpu().numpy()

        try:
            auc = roc_auc_score(y_val_topo, probs)
        except:
            auc = 0.0

        return -auc

    bounds = [(0.01, 1.0)] * len(cams)
    result = differential_evolution(objective, bounds, maxiter=3, popsize=3)
    return result.x / (np.sum(result.x) + 1e-8)





def extract_features(model, volumes, labels, val_images, val_labels):
    model.eval()
    features = []

    for i in tqdm(range(len(volumes)), desc="Extracting features"):
        try:
            vol = volumes[i]  # shape: [D, H, W]
            vol_tensor = torch.tensor(vol, dtype=torch.float32)           # [D, H, W]
            vol_tensor = vol_tensor.unsqueeze(0).permute(0, 2, 3, 1)      # [1, T, H, W]
            vol_tensor = vol_tensor.repeat(3, 1, 1, 1)                     # [3, T, H, W]
            vol_tensor = vol_tensor.unsqueeze(0).to(DEVICE)               # [1, 3, T, H, W]

            with torch.no_grad():
                out = model(vol_tensor)
                class_idx = out.argmax(dim=1).item()

            cams = get_gradcams(model, vol_tensor, class_idx)
            weights = optimize_cam_weights(cams, val_images, val_labels, model)
            fused_cam = fuse_cams(cams, weights)

            segmented, mask = segment_volume(volumes[i], fused_cam)
            gray = segmented.squeeze().astype(np.float32)
            feat = compute_betti_features(gray, mask)

            features.append(np.append(feat, labels[i]))

            if i % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Skipping sample {i}: {type(e).__name__} - {e}")
            continue

    return np.array(features)






def report_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
    cm = confusion_matrix(y_true, y_pred)
    spec = cm[0,0] / (cm[0,0] + cm[0,1]) if cm.shape == (2,2) else 0
    print(f"\n=== Test Metrics ===")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall (Sens): {rec:.4f}")
    print(f"Specificity  : {spec:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"AUC          : {auc:.4f}")
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'specificity': spec, 'f1': f1, 'auc': auc}

def train_topology_mlp(X_train, y_train, X_val, y_val, X_test, y_test):
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    model = TopologyMLP(input_dim, output_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                            torch.tensor(y_train, dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.long)), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                           torch.tensor(y_test, dtype=torch.long)), batch_size=BATCH_SIZE)

    best_val_f1 = 0
    for epoch in range(50):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_probs = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                out = model(xb)
                probs = F.softmax(out, dim=1)
                preds = out.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_probs.extend(probs[:, 1].cpu().numpy() if output_dim > 1 else probs.cpu().numpy())

        val_f1 = f1_score(y_val, val_preds, zero_division=0)
        print(f"Epoch {epoch+1}: Val F1 = {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_topology_mlp.pth"))

    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_topology_mlp.pth")))
    model.eval()
    test_preds, test_probs = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_probs.extend(probs[:, 1].cpu().numpy() if output_dim > 1 else probs.cpu().numpy())

    metrics = report_metrics(y_test, test_preds, test_probs)

    pd.DataFrame(np.column_stack([X_train, y_train])).to_csv(os.path.join(SAVE_DIR, "train_topo.csv"), index=False)
    pd.DataFrame(np.column_stack([X_val, y_val])).to_csv(os.path.join(SAVE_DIR, "val_topo.csv"), index=False)
    pd.DataFrame(np.column_stack([X_test, y_test])).to_csv(os.path.join(SAVE_DIR, "test_topo.csv"), index=False)

    return metrics

def main():
    print("=== TopoCAM: ROI-focused Topological Signatures ===")
    print(f"Using device: {DEVICE}")

    model = build_model(n_classes)
    train_model(model, train_loader, val_loader)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth")))

    train_feats = extract_features(model, X_train, y_train, X_val, y_val)
    val_feats   = extract_features(model, X_val, y_val, X_val, y_val)
    test_feats  = extract_features(model, X_test, y_test, X_val, y_val)

    if train_feats.ndim != 2 or train_feats.shape[1] <= 1:
        raise ValueError("Feature extraction failed — no valid samples returned.")

    X_train_topo, y_train_topo = train_feats[:, :-1], train_feats[:, -1].astype(int)
    X_val_topo, y_val_topo     = val_feats[:, :-1], val_feats[:, -1].astype(int)
    X_test_topo, y_test_topo   = test_feats[:, :-1], test_feats[:, -1].astype(int)

    print("Training MLP on topological features...")
    metrics = train_topology_mlp(X_train_topo, y_train_topo, X_val_topo, y_val_topo, X_test_topo, y_test_topo)

    pd.DataFrame([metrics]).to_csv(os.path.join(SAVE_DIR, "final_results.csv"), index=False)
    print("✅ Pipeline complete. Results saved to:", SAVE_DIR)

if __name__ == "__main__":
    main()


