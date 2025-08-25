import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve
import pandas as pd
from scipy.optimize import differential_evolution

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
N_BINS = 50
HOM_DIMS = [0, 1, 2]
TARGET_SHAPE = (64, 64, 64)

# === Load and preprocess BRATS 2019 data ===
def load_subject(folder_path):
    """Loads 4 modalities and resizes to TARGET_SHAPE."""
    mods = ['flair', 't1', 't1ce', 't2']
    volume = []
    for mod in mods:
        nii_file = [f for f in os.listdir(folder_path) if mod in f.lower() and f.endswith('.nii')]
        if nii_file:
            img = nib.load(os.path.join(folder_path, nii_file[0])).get_fdata()
            factor = [t / s for t, s in zip(TARGET_SHAPE, img.shape)]
            img_resized = zoom(img, factor, order=1)  # linear interpolation
            volume.append(img_resized)
    return np.stack(volume, axis=0) if len(volume) == 4 else None

base_path = "/scratch/09457/bxb210001/BRITIS_2019/MICCAI_BraTS_2019_Data_Training"
classes = {"LGG": 0, "HGG": 1}
X, y = [], []

for label_name, label in classes.items():
    label_folder = os.path.join(base_path, label_name)
    for subject in os.listdir(label_folder):
        sub_path = os.path.join(label_folder, subject)
        if os.path.isdir(sub_path):
            vol = load_subject(sub_path)
            if vol is not None:
                X.append(vol)
                y.append(label)

X = np.array(X)
y = np.array(y)
print(f"Loaded {len(X)} samples. Shape: {X[0].shape} (C,D,H,W)")

# === Split ===
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, stratify=y_train_val, random_state=42)
n_classes = len(np.unique(y))

# === Model ===
class ResNet3DNoInplace(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = r3d_18(weights=R3D_18_Weights.DEFAULT)

        # Modify first conv layer to accept 4 channels instead of 3
        old_conv1 = base.stem[0]
        new_conv1 = nn.Conv3d(
            in_channels=4,  # <-- changed from 3 to 4
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )

        # Initialize new weights by copying pretrained weights and averaging the 4th channel
        with torch.no_grad():
            new_conv1.weight[:, :3] = old_conv1.weight  # copy existing weights
            new_conv1.weight[:, 3:] = old_conv1.weight[:, :1]  # repeat 1st channel

        base.stem[0] = new_conv1  # Replace layer in model
        self.stem = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class MultiLayerFeatureExtractor:
    def __init__(self, model, target_layers):
        self.gradients = {}
        self.activations = {}
        self.handles = [layer.register_forward_hook(self._save_activation(i)) for i, layer in enumerate(target_layers)]
        self.handles += [layer.register_full_backward_hook(self._save_gradient(i)) for i, layer in enumerate(target_layers)]

    def _save_activation(self, idx):
        def hook(module, input, output):
            self.activations[idx] = output.detach()
        return hook

    def _save_gradient(self, idx):
        def hook(module, grad_input, grad_output):
            self.gradients[idx] = grad_output[0].detach()
        return hook

    def get_activations(self): return self.activations
    def get_gradients(self): return self.gradients
    def close(self): [h.remove() for h in self.handles]

from scipy.ndimage import zoom

def compute_weighted_gradcam(model, images, labels, layers, weights, threshold=0.6):
    extractor = MultiLayerFeatureExtractor(model, layers)
    model.eval()

    masked_cams, masks, segments = [], [], []
    weights = torch.tensor(weights, device=device)

    for i in range(len(images)):
        x = torch.tensor(images[i:i+1], dtype=torch.float32).to(device)  # Shape: (1, C, D, H, W)
        y = torch.tensor(labels[i], dtype=torch.long).view(1).to(device)

        model.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        combined_cam = None
        for idx, layer in enumerate(layers):
            act = extractor.get_activations().get(idx)
            grad = extractor.get_gradients().get(idx)
            if act is None or grad is None:
                continue

            pooled_grad = grad.mean(dim=[0, 2, 3, 4], keepdim=True)
            cam = (act * pooled_grad).sum(dim=1, keepdim=True)  # Shape: (1, 1, D, H, W)

            if combined_cam is None:
                combined_cam = torch.zeros_like(cam)

            if cam.shape[2:] != combined_cam.shape[2:]:
                cam = F.interpolate(cam, size=combined_cam.shape[2:], mode='trilinear', align_corners=False)

            combined_cam += weights[idx] * cam

        # Convert CAM to NumPy grayscale
        cam_np = combined_cam.squeeze().detach().cpu().numpy()  # Shape: (D, H, W)
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        cam_np = (cam_np * 255).astype(np.uint8)

        # Resize CAM to match input shape (D, H, W)
        input_shape = images[i].shape[1:]  # (D, H, W) from (C, D, H, W)
        if cam_np.shape != input_shape:
            cam_np = zoom(cam_np, [o / c for o, c in zip(input_shape, cam_np.shape)], order=1)

        # Create binary mask
        binary_mask = (cam_np > int(threshold * 255)).astype(np.uint8)

        # Apply mask to 4D image: (C, D, H, W)
        segmented_img = images[i] * binary_mask[np.newaxis, ...]
        masked_cam = cam_np * binary_mask

        masked_cams.append(masked_cam)
        masks.append(binary_mask)
        segments.append(segmented_img)

    extractor.close()
    return np.array(masked_cams), np.array(masks), np.array(segments)









cp = CubicalPersistence(homology_dimensions=HOM_DIMS, n_jobs=-1)
bc = BettiCurve(n_bins=N_BINS, n_jobs=-1)

def get_betti_vectors(images):
    return bc.fit_transform(cp.fit_transform(images))


def append_pixel_count_to_normalized(betti, imgs):
    norms = np.array([np.count_nonzero(im) for im in imgs])[:, None, None] + 1e-8
    normalized = betti / norms
    counts = np.array([np.count_nonzero(im) for im in imgs])[:, None]
    flat = normalized.reshape(normalized.shape[0], -1)
    return np.concatenate([flat, counts], axis=1)





def compute_metrics(y_true, y_pred, y_prob):
    average = 'binary' if len(np.unique(y_true)) == 2 else 'macro'
    auc = roc_auc_score(y_true, y_prob, multi_class='ovr') if y_prob.ndim == 2 else roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average)
    rec = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    cm = confusion_matrix(y_true, y_pred)
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
    else:
        TP = np.diag(cm)
        FN = np.sum(cm, axis=1) - TP
        FP = np.sum(cm, axis=0) - TP
        TN = np.sum(cm) - (TP + FP + FN)
        sens = np.mean(TP / (TP + FN + 1e-8))
        spec = np.mean(TN / (TN + FP + 1e-8))
    return dict(AUC=auc, Accuracy=acc, Precision=prec, Recall=rec, F1=f1, Sensitivity=sens, Specificity=spec)

def evaluate_mlp(X_train, y_train, X_val, y_val, X_test, y_test):
    clf = MLPClassifier(hidden_layer_sizes=(140, 128, 64), max_iter=500, random_state=42)
    clf.fit(X_train.reshape(len(X_train), -1), y_train.ravel())
    y_pred = clf.predict(X_test.reshape(len(X_test), -1))
    y_prob = clf.predict_proba(X_test.reshape(len(X_test), -1))
    y_score = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob
    return compute_metrics(y_test.ravel(), y_pred, y_score)

# === Run pipeline ===
model = ResNet3DNoInplace(n_classes).to(device)
layers = [model.layer2, model.layer3, model.layer4]



def objective(w):
    # 1. Compute segmentations
    seg_train,_, _ = compute_weighted_gradcam(model, X_train, y_train, layers, w)
    seg_val,_, _ = compute_weighted_gradcam(model, X_val, y_val, layers, w)

    # 2. Extract topological features and process for this model
    bv_train = get_betti_vectors(seg_train)
    bv_val = get_betti_vectors(seg_val)

    # Normalize and append pixel count
    bv_normcount_train = append_pixel_count_to_normalized(bv_train, seg_train)
    bv_normcount_val = append_pixel_count_to_normalized(bv_val, seg_val)

    # 3. Evaluate
    metrics = evaluate_mlp(bv_normcount_train, y_train,
                          bv_normcount_val, y_val,
                          bv_normcount_val, y_val)
    return -metrics['AUC']  # Optimize for validation AUC

# Optimize weights specifically for this model
result_normcount = differential_evolution(
    objective,
    bounds=[(0,1), (0,1), (0,1)],
    maxiter=30,
    popsize=15,
    polish=True
)

best_weights_normcount = result_normcount.x / result_normcount.x.sum()

# Compute final segmentations with optimized weights
_, _,seg_train = compute_weighted_gradcam(model, X_train, y_train, layers, best_weights_normcount)
_, _,seg_val = compute_weighted_gradcam(model, X_val, y_val, layers, best_weights_normcount)
_, _,seg_test = compute_weighted_gradcam(model, X_test, y_test, layers, best_weights_normcount)

# Process features for this model
bv_train = get_betti_vectors(seg_train)
bv_val = get_betti_vectors(seg_val)
bv_test = get_betti_vectors(seg_test)

bv_normcount_train = append_pixel_count_to_normalized(bv_train, seg_train)
bv_normcount_val = append_pixel_count_to_normalized(bv_val, seg_val)
bv_normcount_test = append_pixel_count_to_normalized(bv_test, seg_test)

# Save vector
pd.DataFrame(bv_normcount_train).to_csv("Nonormalization_brats2019_imagemasked_normcount_train.csv", index=False)
pd.DataFrame(bv_normcount_val).to_csv("Nonormalization_brats2019_imagemasked_normcount_val.csv", index=False)
pd.DataFrame(bv_normcount_test).to_csv("Nonormalization_brats2019_imagemasked_normcount_test.csv", index=False)







# Evaluate and save results
results = []
results.append({
    "Model": "MLP_Betti_NormalizedPlusPixelCount",
    **evaluate_mlp(bv_normcount_train, y_train, bv_normcount_val, y_val, bv_normcount_test, y_test)
})

pd.DataFrame(results).to_csv("Nonormalization_brats2019_normcount_results.csv", index=False)


