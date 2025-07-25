import time

t_script_start = time.time()

# Imports & Settings
print("Imports & Settings")
t0 = time.time()

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    log_loss,
    roc_auc_score,
    roc_curve,
)

RANDOM_STATE = 42

print(f" → done in {time.time() - t0:.1f}s")


# Load & Inspect Data
print("\n Load & Inspect Data")
t0 = time.time()

df = pd.read_csv("./Darknet.CSV", engine="python", on_bad_lines="skip")

print(" Raw shape:", df.shape)
print(" Sample labels:\n", df.filter(like="Label").head())
print(" Label dist:\n", df["Label"].value_counts())
print(" Label.1 dist:\n", df["Label.1"].value_counts())

print(f" → done in {time.time() - t0:.1f}s")

# Data Preprocessing
print("\n Preprocessing")
t0 = time.time()

# Drop string cols
df["Label.1"] = df["Label.1"].str.strip().str.title()
to_drop = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]
df = df.drop(columns=[c for c in to_drop if c in df.columns])
df = df.replace([np.inf, -np.inf], np.nan).dropna()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X_df = df[numeric_cols].astype(np.float32)
X = X_df.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f" X shape: {X_scaled.shape}")
print(f" → done in {time.time() - t0:.1f}s")

# PCA & t-SNE Visualization
print("\n PCA & t-SNE scatter")
t0 = time.time()

# PCA
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

le_temp = LabelEncoder()
y_temp = np.array(le_temp.fit_transform(df["Label.1"]))

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_temp, cmap="tab10", s=3, alpha=0.5)
plt.title("PCA Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Class")
plt.tight_layout()
plt.savefig("pca_scatter.png")
plt.close()

# t-SNE (on a random subset)
subset = np.random.RandomState(RANDOM_STATE).choice(
    X_scaled.shape[0], size=min(5000, X_scaled.shape[0]), replace=False
)

tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
X_tsne = tsne.fit_transform(X_scaled[subset])

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_temp[subset], cmap="tab20", s=3, alpha=0.6)
plt.title("t-SNE Projection (5k samples)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.colorbar(label="Class")
plt.tight_layout()
plt.savefig("tsne_scatter.png")
plt.close()

print(f" → done in {time.time() - t0:.1f}s")

# Loop over both labels
for label_col in ["Label", "Label.1"]:
    print(f"\n===== Processing `{label_col}` =====")
    t_label_start = time.time()

    t0 = time.time()
    y = df[label_col]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_enc, test_size=0.20, random_state=RANDOM_STATE, stratify=y_enc
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=RANDOM_STATE, stratify=y_tr
    )
    print(f" Data split done in {time.time()-t0:.1f}s")

    t0 = time.time()
    base = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    param_grid = {
        "learning_rate": [0.05, 0.1, 0.2],
        "max_iter": [100, 200],
        "max_depth": [None, 10],
    }
    gs = GridSearchCV(base, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
    gs.fit(X_tr, y_tr)
    clf = gs.best_estimator_
    print(f" GridSearch+train done in {time.time()-t0:.1f}s → best: {gs.best_params_}")

    # Validation Metrics
    t0 = time.time()
    y_val_proba = clf.predict_proba(X_val)
    print(" VAL log-loss :", log_loss(y_val, y_val_proba))
    print(" VAL accuracy :", accuracy_score(y_val, clf.predict(X_val)))
    print(" VAL ROC-AUC  :", roc_auc_score(y_val, y_val_proba, multi_class="ovo"))
    print(f" Validation eval in {time.time()-t0:.1f}s")

    # Test Metrics
    t0 = time.time()
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)

    print(" Test accuracy        :", accuracy_score(y_te, y_pred))
    print(
        " Classification report:\n",
        classification_report(y_te, y_pred, target_names=le.classes_),
    )
    cm = confusion_matrix(y_te, y_pred)
    print(" Test log-loss        :", log_loss(y_te, y_proba))
    print(" Test ROC-AUC         :", roc_auc_score(y_te, y_proba, multi_class="ovo"))
    print(f" Test eval in {time.time()-t0:.1f}s")

    # ROC Curves (one-vs-rest)
    t0 = time.time()
    y_bin = np.array(label_binarize(y_te, classes=np.arange(len(le.classes_))))
    fprs, tprs, aucs = {}, {}, {}
    for i in range(len(le.classes_)):
        fprs[i], tprs[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        aucs[i] = roc_auc_score(y_bin[:, i], y_proba[:, i])

    # Sorted ROC Plot
    sorted_auc_items = sorted(aucs.items(), key=lambda x: x[1], reverse=True)
    plt.figure(figsize=(8, 6))
    for cls_idx, auc in sorted_auc_items:
        plt.plot(
            fprs[cls_idx],
            tprs[cls_idx],
            lw=1,
            label=f"{le.classes_[cls_idx]} (AUC={auc:.2f})",
        )
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title(f"ROC Curves ({label_col})", fontsize=14)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"roc_{label_col}.png")
    plt.close()
    print(f" ROC curves done in {time.time()-t0:.1f}s")

    # Save Artifacts
    t0 = time.time()
    joblib.dump(clf, f"hgb_model_{label_col}.joblib")
    joblib.dump(le, f"label_encoder_{label_col}.joblib")

    # Confusion Matrix
    plt.figure(figsize=(9, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="mako",
        cbar=True,
        xticklabels=le.classes_.tolist(),
        yticklabels=le.classes_.tolist(),
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 10},
    )
    plt.title(f"Confusion Matrix ({label_col})", fontsize=14)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"cm_{label_col}.png")
    plt.close()

    print(f" Artifacts saved in {time.time()-t0:.1f}s")
    print(f" Total for {label_col}: {time.time()-t_label_start:.1f}s")

print(f"\nAll done in {time.time() - t_script_start:.1f}s")
