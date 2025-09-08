import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.optimize import differential_evolution
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm
import gc

# === CONFIG ===
ATA_NPZ = "/work/09457/bxb210001/ls6/MEDMNIST_NOV/breastmnist_224.npz"
TARGET_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
THRESHOLD = 0.6
HOM_DIMS = [0, 1]
N_BINS = 50
SAVE_DIR = "sample_breast_cancer_features"
os.makedirs(SAVE_DIR, exist_ok=True)

# === LOAD DATA ===
data = np.load(ATA_NPZ)
X_train, y_train = data["train_images"], data["train_labels"].astype(int)
X_val, y_val = data["val_images"], data["val_labels"].astype(int)
X_test, y_test = data["test_images"], data["test_labels"].astype(int)

# === SUBSAMPLE TO 5 PER CLASS (10 TOTAL) PER SET ===
print("Loading data...")
data = np.load(DATA_NPZ)
X_train, y_train = data["train_images"], data["train_labels"]
X_val, y_val = data["val_images"], data["val_labels"]
X_test, y_test = data["test_images"], data["test_labels"]

y_train = y_train.reshape(-1).astype(int)
y_val = y_val.reshape(-1).astype(int)
y_test = y_test.reshape(-1).astype(int)
n_classes = int(np.unique(np.concatenate([y_train, y_val, y_test])).size)



# Resize and normalize
def preprocess(images):
    resized = np.stack([np.resize(img, TARGET_SIZE) for img in images])
    return resized

X_train = preprocess(X_train)
X_val = preprocess(X_val)
X_test = preprocess(X_test)

# === DATASET ===
class BreastDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToTensor(),                      # [1, H, W]
            transforms.Resize(TARGET_SIZE),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # [3, H, W]
        ])

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = self.transform(self.images[idx])
        return img, self.labels[idx]

train_loader = DataLoader(BreastDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(BreastDataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(BreastDataset(X_test, y_test), batch_size=BATCH_SIZE)

# === MODEL ===
def build_model(n_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
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
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
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

def segment_image(image, fused_cam, tau=THRESHOLD):
    mask = (fused_cam >= tau).astype(np.uint8)
    return image * mask, mask

# === TOPOLOGY ===
cp = CubicalPersistence(homology_dimensions=HOM_DIMS, n_jobs=-1)
bc = BettiCurve(n_bins=N_BINS, n_jobs=-1)

def compute_betti_features(gray_image, mask):
    nonzero = np.count_nonzero(mask)
    if nonzero == 0:
        raise ValueError("Empty mask")
    di = cp.fit_transform([gray_image])
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


def extract_features(model, images, labels, val_images, val_labels):
    model.eval()
    features = []

    for i in tqdm(range(len(images)), desc="Extracting features"):
        try:
            img = images[i]  # shape: [H, W]
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
            img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=TARGET_SIZE, mode='bilinear')  # [1, 1, H, W]
            img_tensor = img_tensor.repeat(1, 3, 1, 1).to(DEVICE)  # [1, 3, H, W]

            with torch.no_grad():
                out = model(img_tensor)
                class_idx = out.argmax(dim=1).item()

            cams = get_gradcams(model, img_tensor, class_idx)
            weights = optimize_cam_weights(cams, val_images, val_labels, model)
            fused_cam = fuse_cams(cams, weights)

            segmented, mask = segment_image(img, fused_cam)
            gray = segmented.astype(np.float32)
            feat = compute_betti_features(gray, mask)

            features.append(np.append(feat, labels[i]))

            if i % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f" Skipping sample {i}: {type(e).__name__} - {e}")
            continue

    return np.array(features)

def optimize_cam_weights(cams, val_images, val_labels, model):
    def objective(weights):
        weights = np.array(weights)
        weights /= weights.sum() + 1e-8
        val_feats = []

        for img, label in zip(val_images, val_labels):
            try:
                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
                img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=TARGET_SIZE, mode='bilinear')  # [1, 1, H, W]
                img_tensor = img_tensor.repeat(1, 3, 1, 1).to(DEVICE)  # [1, 3, H, W]

                with torch.no_grad():
                    out = model(img_tensor)
                    class_idx = out.argmax(dim=1).item()

                val_cams = get_gradcams(model, img_tensor, class_idx)
                fused = fuse_cams(val_cams, weights)
                segmented, mask = segment_image(img, fused)
                gray = segmented.astype(np.float32)
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
        for _ in range(100):
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
    result = differential_evolution(objective, bounds, maxiter=50, popsize=15)
    return result.x / (np.sum(result.x) + 1e-8)


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

    # Build and train CNN
    model = build_model(n_classes=2)
    train_model(model, train_loader, val_loader)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth")))

    # Extract topological features
    train_feats = extract_features(model, X_train, y_train, X_val, y_val)
    val_feats   = extract_features(model, X_val, y_val, X_val, y_val)
    test_feats  = extract_features(model, X_test, y_test, X_val, y_val)

    # Validate feature shape
    if train_feats.ndim != 2 or train_feats.shape[1] <= 1:
        raise ValueError("Feature extraction failed — no valid samples returned.")

    # Split features and labels
    X_train_topo, y_train_topo = train_feats[:, :-1], train_feats[:, -1].astype(int)
    X_val_topo, y_val_topo     = val_feats[:, :-1], val_feats[:, -1].astype(int)
    X_test_topo, y_test_topo   = test_feats[:, :-1], test_feats[:, -1].astype(int)

    # Train MLP classifier
    print("Training MLP on topological features...")
    metrics = train_topology_mlp(X_train_topo, y_train_topo, X_val_topo, y_val_topo, X_test_topo, y_test_topo)

    # Save final metrics
    pd.DataFrame([metrics]).to_csv(os.path.join(SAVE_DIR, "final_results.csv"), index=False)
    print("✅ Pipeline complete. Results saved to:", SAVE_DIR)


if __name__ == "__main__":
    main()
