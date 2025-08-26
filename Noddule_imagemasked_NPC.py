#  pipeline with masked Grad-CAM and Betti vector saving
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.ndimage import zoom
# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
N_BINS = 50
HOM_DIMS = [0, 1, 2]

# Load dataset
data = np.load("/work/09457/bxb210001/ls6/3D_UPDATED_project/nodulemnist3d_64.npz")
X_train, y_train = data["train_images"], data["train_labels"]
X_val, y_val = data["val_images"], data["val_labels"]
X_test, y_test = data["test_images"], data["test_labels"]
n_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))

# Model definition
class ResNet3DNoInplace(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = r3d_18(weights=R3D_18_Weights.DEFAULT)
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
        self.handles = []
        for i, layer in enumerate(target_layers):
            self.handles.append(layer.register_forward_hook(self._save_activation(i)))
            self.handles.append(layer.register_full_backward_hook(self._save_gradient(i)))

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
    def close(self):
        for h in self.handles:
            h.remove()

def compute_weighted_gradcam(model, images, labels, layers, weights, threshold=0.6):
    extractor = MultiLayerFeatureExtractor(model, layers)
    model.eval()

    cams, masks, segments, masked_cams = [], [], [], []
    weights = torch.tensor(weights, device=device)

    for i in range(len(images)):
        original_img = images[i]  # Shape: (64, 64, 64)
        img_tensor = torch.tensor(original_img[np.newaxis, np.newaxis], dtype=torch.float32).repeat(1, 3, 1, 1, 1).to(device)
        label_tensor = torch.tensor(int(labels[i]), dtype=torch.long, device=device).view(1)

        model.zero_grad()
        logits = model(img_tensor)
        loss = F.cross_entropy(logits, label_tensor)
        loss.backward()

        combined_cam = None
        for idx, layer in enumerate(layers):
            act = extractor.get_activations().get(idx)
            grad = extractor.get_gradients().get(idx)
            if act is None or grad is None:
                continue

            pooled_grad = grad.mean(dim=[0, 2, 3, 4], keepdim=True)
            cam = (act * pooled_grad).sum(dim=1, keepdim=True)  # Shape: (1,1,D,H,W)

            if combined_cam is None:
                combined_cam = torch.zeros_like(cam)

            if cam.shape[2:] != combined_cam.shape[2:]:
                cam = F.interpolate(cam, size=combined_cam.shape[2:], mode='trilinear', align_corners=False)

            combined_cam += weights[idx] * cam

        # === Convert to numpy ===
        cam_np = combined_cam.squeeze().detach().cpu().numpy()

        # === Normalize to [0, 255] grayscale ===
        cam_np = cam_np - cam_np.min()
        cam_np = (cam_np / (cam_np.max() + 1e-8)) * 255.0
        cam_np = cam_np.astype(np.uint8)

        # === Resize CAM to match original image shape if needed ===
        if cam_np.shape != original_img.shape:
            cam_np = zoom(cam_np, [o / c for o, c in zip(original_img.shape, cam_np.shape)], order=1)

        # === Binary mask + apply ===
        binary_mask = (cam_np > int(threshold * 255)).astype(np.uint8)
        segmented_img = (original_img * binary_mask).astype(original_img.dtype)
        masked_cam = (cam_np * binary_mask).astype(np.uint8)

        cams.append(cam_np)
        masks.append(binary_mask)
        segments.append(segmented_img)
        masked_cams.append(masked_cam)

    extractor.close()
    return np.array(masked_cams), np.array(masks), np.array(segments)


def evaluate_mlp(X_train, y_train, X_test_1, y_test_1):
    clf = MLPClassifier(hidden_layer_sizes=(140, 100, 64), max_iter=500, random_state=42)
    clf.fit(X_train.reshape(len(X_train), -1), y_train.ravel())
    y_pred = clf.predict(X_test_1.reshape(len(X_test_1), -1))
    y_prob = clf.predict_proba(X_test_1.reshape(len(X_test_1), -1))
    y_score = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob
    return compute_metrics(y_test_1.ravel(), y_pred, y_score)


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


# Initialize model
model = ResNet3DNoInplace(n_classes).to(device)
layers = [model.layer2, model.layer3, model.layer4]

# Define objective function for weight optimization
def objective(w):
    seg_train,_,_= compute_weighted_gradcam(model, X_train, y_train, layers, w)
    seg_val,_,_= compute_weighted_gradcam(model, X_val, y_val, layers, w)
    bv_train = get_betti_vectors(seg_train)
    bv_val = get_betti_vectors(seg_val)
    bv_normcount_train = append_pixel_count_to_normalized(bv_train, seg_train)
    bv_normcount_val = append_pixel_count_to_normalized(bv_val, seg_val)
    metrics = evaluate_mlp(bv_normcount_train, y_train,  bv_normcount_val, y_val)
    return -metrics['AUC']

# Optimize weights
result = differential_evolution(objective, bounds=[(0,1)]*3, maxiter=30, popsize=15, polish=True)
best_weights = result.x / result.x.sum()

# Generating class-wise segmented images for the test, train, and validation set
_, _,seg_train = compute_weighted_gradcam(model, X_train, y_train, layers, best_weights)
_, _,seg_val = compute_weighted_gradcam(model, X_val, y_val, layers, best_weights)
_, _,seg_test = compute_weighted_gradcam(model, X_test, y_test, layers, best_weights)

bv_train = get_betti_vectors(seg_train)
bv_val = get_betti_vectors(seg_val)
bv_test = get_betti_vectors(seg_test)

bv_normcount_train = append_pixel_count_to_normalized(bv_train, seg_train)
bv_normcount_val = append_pixel_count_to_normalized(bv_val, seg_val)
bv_normcount_test = append_pixel_count_to_normalized(bv_test, seg_test)

# Save vector
pd.DataFrame(bv_normcount_train).to_csv("nodule_imagemasked_normcount_train.csv", index=False)
pd.DataFrame(bv_normcount_val).to_csv("nodule_imagemasked_normcount_val.csv", index=False)
pd.DataFrame(bv_normcount_test).to_csv("nodule_imagemasked_normcount_test.csv", index=False)



