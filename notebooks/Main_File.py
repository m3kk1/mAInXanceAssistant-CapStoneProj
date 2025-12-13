#!/usr/bin/env python
# coding: utf-8

# 
# # Capstone Project Overview
# 

# In[ ]:


#### Capstone Project: Predicting Technician Actions Using Failures based on Work Orders History ####

# This notebook walks through the process of:
# Cleaning real-world technician notes
# Labeling word clusters for similar behaviors from text
# Training ML models to predict technician actions based on past work order descriptions
# Goal: Assist maintenance teams with smart fix suggestions based on past patterns.

### WORKFLOW ###

# 1. Data management - Load and clean data (Drop rows with incomplete or irrelevant info)
# 2. System Labeling - NLP
# 3. Dimensionality Reduction (PCA/TSNE/UMAP for visuals)
# 4. Train/Test Split (X = description, y = solution/note)
# 5. Model Training: DT, RF, NN
# 6. Evaluation + Narrative Generation


# # **1. Data Management**

# # Imports & Installs

# In[ ]:


# --- One-time setup 
import sys, subprocess

def pip_install(pkg):
    try:
        __import__(pkg.split("[")[0].replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

# (pip install)
pip_install("sentence-transformers")
pip_install("fuzzywuzzy[speedup]")
pip_install("groq")        
pip_install("openai")      


# In[ ]:


# ===============================
# Core
# ===============================
subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn", "-q"])
import os, re, warnings
import numpy as np
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "-q"])
import pandas as pd
import joblib
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ===============================
# NLP / Vectorization
# ===============================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# embedding baseline (semantic retrieval)
from sentence_transformers import SentenceTransformer

# fuzzy matching utility 
from fuzzywuzzy import fuzz

# ===============================
# Modeling (classification scope)
# ===============================
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# ===============================
# Metrics
# ===============================
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
)

# ===============================
# Visualization
# ===============================
import matplotlib.pyplot as plt
import seaborn as sns

# Dimensionality-reduction for visuals 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap  # This will now work after installing umap-learn

# ===============================
# LLM Clients (for narrative layer)
# ===============================
from groq import Groq        
from openai import OpenAI


# In[ ]:


subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "python-Levenshtein", "-q"])


# # Cleaning Data

# In[ ]:


import os
# Load CSV from the repository root directory
csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cleaned_work_orders.csv")
df = pd.read_csv(csv_path)

df = df.dropna(how='all')

# Drop noise in Data
df = df[~df['Text'].str.contains("completed", case=False, na=False)]
df = df[~df['Text'].str.contains("complete", case=False, na=False)]
df = df[~df['Text'].str.contains("mike", case=False, na=False)]
df = df[~df['Text'].str.contains("mike's", case=False, na=False)]
df = df[~df['Text'].str.contains("odd", case=False, na=False)]

df = df.dropna(subset=['Description', 'Text']).drop_duplicates()

# Keep only rows where WO No. is numeric
df = df[df['WO No.'].astype(str).str.match(r'^\d+$')]

# Strip spaces and standardize text fields
df['WO No.'] = df['WO No.'].astype(str).str.strip()
df['Description'] = df['Description'].astype(str).str.strip()
df['Text'] = df['Text'].astype(str).str.strip()

# Clean Description
df['Description_cleaned'] = (
    df['Description']
    .fillna("")
    .str.lower()
    .str.replace(r'[^\w\s]', '', regex=True)  # Remove punctuation
    .str.replace(r'\d+', '', regex=True)      # Remove digits
    .str.strip()
)

# Clean Technician Text Notes
df['Text_cleaned'] = (
    df['Text']
    .fillna("")
    .str.lower()
    .str.replace(r'[^\w\s]', '', regex=True)
    .str.replace(r'\d+', '', regex=True)
    .str.strip()
)

# Keep rows where both sides exist and are not trivial
df = df[['Description_cleaned', 'Text_cleaned']].dropna()
df = df[df['Description_cleaned'].str.strip() != ""]
df = df[df['Text_cleaned'].str.strip() != ""]

# Filter out rows with completely blank descriptions and notes
df = df[~((df['Description_cleaned'].str.strip() == "") & (df['Text_cleaned'].str.strip() == ""))]

# Ensure valid technician notes
df = df[df['Text_cleaned'].notna() & (df['Text_cleaned'].str.strip() != "") & (df['Text_cleaned'].str.lower().str.strip() != "nan")]

# Show data sample for verification
print(df[['Description_cleaned', 'Text_cleaned']].head())


# # **2. System Labeling**

# In[ ]:


# Map free-text notes to a small, useful label set.

# --- 1) Primary rule-based mapping ---
ACTION_PATTERNS = [
    ("Replace Part",   r"\b(replace|replaced|swap|swapped|install(ed)?)\b.*\b(bearing|motor|belt|gear|fuse|sensor|valve|hose|coupling|chain|switch|roller|pulley|seal)\b"),
    ("Tighten/Adjust", r"\b(tighten|tightened|adjust|adjusted|align|aligned|re-seat|reseat|calibrate|calibrated|reposition|realign(ed)?)\b"),
    ("Clean/Clear",    r"\b(clean|cleaned|clear|cleared|remove|removed)\b.*\b(debris|dust|jam|blockage|clog)\b|\b(cleaned|cleared)\b"),
    ("Refill/Top Off", r"\b(add|added|refill|refilled|top\s?off)\b.*\b(oil|fluid|grease|lub(e|ricant)|coolant)\b"),
    ("Electrical Fix", r"\b(replace|replaced|reset|rewire|wire(d)?|reconnect|connector|contactor|breaker|fuse|vfd|plc|relay)\b"),
    ("Hydraulic/Pneumatic Fix", r"\b(hose|cylinder|solenoid|regulator|air line|hydraulic|pneumatic)\b.*\b(repair|replace|fixed|leak|leaking)\b"),
    ("Reset/Power Cycle", r"\b(reset|power.?cycle|cycled|restart|reboot|restarted)\b"),
    ("Inspection/Test Only", r"\b(inspect|inspected|tested|verify|verified|checked)\b(?!.*replace|.*repair|.*fix)"),
    ("Other", r".*")
]

import re

def to_response_label(text: str) -> str:
    t = " " + str(text).lower() + " "
    for label, pat in ACTION_PATTERNS:
        if re.search(pat, t):
            return label
    return "Other"

df['Response_Label'] = df['Text_cleaned'].apply(to_response_label)


# --- 2) Secondary re-mapping for rows still tagged "Other" ---
# Uses precise bigrams first, then high-signal unigrams, mapped into EXISTING labels only.

BIGRAM_MAP = [
    # Installation / replacement
    (["installed new", "fabricated new", "changed torch", "removed broken"], "Replace Part"),

    # Pneumatic / hydraulic
    (["air line", "air pressure", "air leak", "solenoid valve", "foot pedal"], "Hydraulic/Pneumatic Fix"),

    # Electrical
    (["limit switch", "power supply", "light curtain", "repaired wiring"], "Electrical Fix"),

    # Inspection / test outcomes
    (["started working", "went away", "working ok", "worked fine", "ran fine"], "Inspection/Test Only"),

    # Mechanical tighten/adjust
    (["came loose", "took apart"], "Tighten/Adjust"),
]

UNIGRAM_MAP = [
    # Installation / replacement
    (["installed", "install", "changed", "removed", "new"], "Replace Part"),

    # Pneumatic / hydraulic
    (["air", "line", "hose", "pump", "cylinder", "solenoid", "regulator"], "Hydraulic/Pneumatic Fix"),

    # Electrical
    (["switch", "wiring", "controller", "power", "fuse"], "Electrical Fix"),

    # Inspection / test / ambiguous OK
    (["found", "checked", "ok", "not working", "problem"], "Inspection/Test Only"),

    # Mechanical adjust
    (["loose", "aligned", "adjust", "tighten", "tightened"], "Tighten/Adjust"),
]

def remap_other_label(note: str) -> str:
    t = str(note).lower()

    # 1) Bigram priority (exact substring search for speed/clarity)
    for phrases, mapped in BIGRAM_MAP:
        if any(p in t for p in phrases):
            return mapped

    # 2) Unigram fallbacks
    for terms, mapped in UNIGRAM_MAP:
        if any(w in t for w in terms):
            return mapped

    return "Other"

mask_other = (df["Response_Label"] == "Other")
df.loc[mask_other, "Response_Label"] = df.loc[mask_other, "Text_cleaned"].apply(remap_other_label)


# --- 3) Collapse tiny labels again (keeps classes trainable) ---
min_count = 15
vc = df['Response_Label'].value_counts()
valid = vc[vc >= min_count].index
df.loc[~df['Response_Label'].isin(valid), 'Response_Label'] = "Other"

# new distribution to confirm "Other" 
print("Label distribution AFTER re-map:\n", df['Response_Label'].value_counts())



# # **3. Dimensionality Reduction (PCA/TSNE/UMAP for visuals)**

# # PCA

# In[ ]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer # Ensure TfidfVectorizer is imported if not globally available

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = tfidf.fit_transform(df['Text_cleaned'].fillna(''))
y = df['Response_Label']

# Use sklearn's PCA explicitly
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_tfidf.toarray())

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='tab10')
plt.title("PCA - TF-IDF Text_cleaned")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title='Text', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# # TSNE

# In[ ]:


tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
X_tsne = tsne.fit_transform(X_tfidf.toarray())

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='tab10')
plt.title("t-SNE - TF-IDF Response_Label")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title='Response', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# # UMAP

# In[ ]:


reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_tfidf)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette='tab10')
plt.title("UMAP - TF-IDF Response_Label")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title='Response', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[ ]:


# ==== Save PCA / t-SNE / UMAP artifacts & plots ====
import numpy as np, pandas as pd, joblib, time
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# y labels for coloring
try:
    y_labels = y_train_text.astype(str).reset_index(drop=True)
except Exception:
    y_labels = None

# Helper: convert to dense float32 safely
def to_dense_float32(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    return X

# Use the TRAIN matrix you clustered/classified on
assert 'X_train' in globals(), "X_train_vec not found. Provide your train feature matrix."
Xmat_full = X_train
n = Xmat_full.shape[0]

# Subsample for heavy methods to keep plots readable/fast
MAX_TSNE = 5000
MAX_UMAP = 15000
rng = np.random.RandomState(42)

def subsample(X, y, k):
    if X.shape[0] <= k:
        return X, y, np.arange(X.shape[0])
    idx = rng.choice(X.shape[0], k, replace=False)
    if y is not None:
        return X[idx], y.iloc[idx].reset_index(drop=True), idx
    return X[idx], None, idx

# ---------- PCA (fast, saves model + coords + plot) ----------
print("Running PCA(2)…")
Xp = to_dense_float32(Xmat_full)
pca = PCA(n_components=2, random_state=42)
P_pca = pca.fit_transform(Xp)

# Plot PCA
plt.figure(figsize=(7,6))
if y_labels is not None:
    # basic coloring by label
    for lab in pd.Series(y_labels).unique():
        m = (y_labels == lab).values
        plt.scatter(P_pca[m,0], P_pca[m,1], s=6, alpha=0.7, label=str(lab))
    plt.legend(markerscale=3, fontsize=8, frameon=False)
else:
    plt.scatter(P_pca[:,0], P_pca[:,1], s=6, alpha=0.7)
plt.title("PCA (2D)")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout(); plt.show()

# ---------- t-SNE (coords + plot; no reusable model) ----------
print("Running t-SNE(2)… (subsample if needed)")
Xts, yts, idx_ts = subsample(Xp, y_labels, MAX_TSNE)

# Pick a sane perplexity relative to sample size
perp = max(5, min(30, (Xts.shape[0] - 1) // 3))
tsne = TSNE(n_components=2, perplexity=perp, learning_rate="auto",
            init="pca", n_iter=1000, random_state=42, verbose=0)
P_tsne = tsne.fit_transform(Xts)

# ---------- UMAP (saves model + coords + plot, if available) ----------
print("Running UMAP(2)… (subsample if needed)")
HAS_UMAP = False
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except Exception:
    pass

if HAS_UMAP:
    Xum, yum, idx_um = subsample(Xp, y_labels, MAX_UMAP)
    um = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        verbose=False,
    )
    P_umap = um.fit_transform(Xum)

    # Save UMAP model & coords
    joblib.dump(um, SAVE_DIR / "umap_2d.joblib")
    pd.DataFrame({
        "umap_x": P_umap[:,0],
        "umap_y": P_umap[:,1],
        "label": yum if yum is not None else None,
        "orig_index": idx_um
    }).to_csv(SAVE_DIR / "umap_2d_coords.csv", index=False)

    # Plot UMAP
    plt.figure(figsize=(7,6))
    if yum is not None:
        for lab in pd.Series(yum).unique():
            m = (yum == lab).values
            plt.scatter(P_umap[m,0], P_umap[m,1], s=6, alpha=0.8, label=str(lab))
        plt.legend(markerscale=3, fontsize=8, frameon=False)
    else:
        plt.scatter(P_umap[:,0], P_umap[:,1], s=6, alpha=0.8)
    plt.title(f"UMAP (2D, n={Xum.shape[0]})")
    plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout(); plt.show()
else:
    print("UMAP not installed; skipped. Install with: pip install umap-learn")

print("Done. Files written to:", SAVE_DIR.resolve())


# # Sentence Transformer
# 

# In[ ]:


# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define features and labels
X = model.encode(df['Description_cleaned'].tolist(), show_progress_bar=True)
y = df['Response_Label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# In[ ]:


# Set up features and labels
X_embed = model.encode(df['Description_cleaned'].tolist(), show_progress_bar=True)
X = X_embed
y = df['Response_Label']


# # **4. Split Train & Test**

# In[ ]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from collections import Counter
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize 
vectorizer = TfidfVectorizer(stop_words='english', max_features=8000, min_df=3)
X = vectorizer.fit_transform(df['Description_cleaned'].astype(str))
y = df['Response_Label'].astype(str)

# Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Dimensionality reduction 
svd = TruncatedSVD(n_components=150, random_state=42)
X_dense = svd.fit_transform(X)  

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_dense, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# SMOTE on dense features
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Original class counts:", Counter(y))
print("Train (encoded) before:", Counter(y_train))
print("Train (encoded) after :", Counter(y_train_res))


# # Training Prep

# In[ ]:


# Convert to numpy for sklearn
X_embed_np = np.array(X_embed)

# Prepare target labels (supervised)
y = df['Response_Label'].reset_index(drop=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_embed_np, y, test_size=0.2, stratify=y, random_state=42)

# Train classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# 8. Evaluate
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# # **5. Modeling**

# # ML Pipeline Cross-Validation

# In[ ]:


# ==== Lean model comparison (TF-IDF inside pipelines) ====
import numpy as np, pandas as pd
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ---- XGBoost (will skip if not installed) ----
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ===== Data =====
X_text = df["Description_cleaned"].astype(str).values
y_text = df["Response_Label"].astype(str).values

# Encode labels once (needed for XGBoost; OK for others too)
le = LabelEncoder()
y = le.fit_transform(y_text)

# ===== CV setup =====
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
tfidf = TfidfVectorizer(stop_words="english", max_features=6000, ngram_range=(1,2), min_df=3)

# ===== Pipelines =====
models = OrderedDict()

models["NaiveBayes"] = Pipeline([
    ("tfidf", tfidf),
    ("clf", MultinomialNB(alpha=0.5))
])

models["LogReg"] = Pipeline([
    ("tfidf", tfidf),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))
])

models["DecisionTree"] = Pipeline([
    ("tfidf", tfidf),
    ("clf", DecisionTreeClassifier(max_depth=18, min_samples_leaf=3, random_state=42))
])

models["RandomForest"] = Pipeline([
    ("tfidf", tfidf),
    ("clf", RandomForestClassifier(
        n_estimators=200, max_depth=18, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42))
])

if HAS_XGB:
    models["XGBoost"] = Pipeline([
        ("tfidf", tfidf),
        ("clf", xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(np.unique(y)),
            n_estimators=400, max_depth=8, learning_rate=0.2,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", eval_metric="mlogloss",
            n_jobs=-1, random_state=42))
    ])

# ===== Cross-validation (macro-F1 + accuracy) =====
rows = []
for name, pipe in models.items():
    f1 = cross_val_score(pipe, X_text, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    acc = cross_val_score(pipe, X_text, y, cv=cv, scoring="accuracy", n_jobs=-1)
    rows.append({
        "model": name,
        "cv_f1_macro_mean": f1.mean(),
        "cv_f1_macro_std": f1.std(),
        "cv_accuracy_mean": acc.mean(),
        "cv_accuracy_std": acc.std()
    })

res = pd.DataFrame(rows).sort_values("cv_f1_macro_mean", ascending=False).reset_index(drop=True)
print("=== Cross-validated results (3-fold, stratified) ===")
print(res)

# =====  Fit the top model on a proper train/test split and report test metrics =====
top_model_name = res.iloc[0]["model"]
print(f"\nTraining top model on a hold-out split: {top_model_name}")

# split raw text (no leakage), then fit pipeline on train, evaluate on test
X_tr_text, X_te_text, y_tr, y_te = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

pipe = models[top_model_name]
pipe.fit(X_tr_text, y_tr)

y_pred = pipe.predict(X_te_text)
print("\n=== Test set metrics (hold-out) ===")
print("Accuracy:", f"{accuracy_score(y_te, y_pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_te, y_pred, target_names=le.classes_, digits=3))


# # Naive Bayes

# In[ ]:


# === Naive Bayes (TF-IDF)  ===
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
import joblib


# 1) Train pipeline (TF-IDF -> Naive Bayes)
pipe_nb = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("clf", MultinomialNB(alpha=0.5))
])
pipe_nb.fit(X_train_text, y_train)

# 2) Predictions
y_nb = pipe_nb.predict(X_test_text)

print("\n=== Naive Bayes (TF-IDF) — Test ===")
print(classification_report(y_test, y_nb, digits=3))


# Per-class PR/Recall/F1 bars (if helper exists)
try:
    plot_prf_bars(prf_table(y_test, y_nb), "Naive Bayes — Precision/Recall/F1")
except Exception:
    pass  # skip gracefully if helpers not defined

# 3) Macro Precision–Recall curve (model has predict_proba)
try:
    macro_pr_curve(pipe_nb, X_test_text, y_test)
except Exception:
    # minimal fallback: macro AP only
    classes = sorted(np.unique(y_test))
    Y = label_binarize(y_test, classes=classes)
    P = pipe_nb.predict_proba(X_test_text)
    aps = [average_precision_score(Y[:,k], P[:,k]) for k in range(Y.shape[1])]
    print(f"Macro AP (fallback): {np.mean(aps):.3f}")

# 4) Multiclass ROC — macro & micro (clean, no per-class clutter)
classes = sorted(np.unique(y_test))
Y = label_binarize(y_test, classes=classes)         # shape: N x K
P = pipe_nb.predict_proba(X_test_text)              # shape: N x K

# Micro-average ROC
fpr_micro, tpr_micro, _ = roc_curve(Y.ravel(), P.ravel())
auc_micro = auc(fpr_micro, tpr_micro)

# Macro-average ROC
fpr = dict(); tpr = dict(); roc_auc = dict()
for i in range(Y.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(Y[:, i], P[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Aggregate macro
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(Y.shape[1])]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(Y.shape[1]):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= Y.shape[1]
auc_macro = auc(all_fpr, mean_tpr)

plt.figure(figsize=(6.5, 4.5), dpi=150)
plt.plot(fpr_micro, tpr_micro, lw=2, label=f"micro-avg ROC (AUC={auc_micro:.3f})")
plt.plot(all_fpr,   mean_tpr,  lw=2, label=f"macro-avg ROC (AUC={auc_macro:.3f})")
plt.plot([0,1], [0,1], "k--", lw=1)
plt.title("Naive Bayes — ROC (micro & macro)")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.xlim(0,1); plt.ylim(0,1.01); plt.legend(loc="lower right"); plt.tight_layout(); plt.show()


# # Random Forest Classifier

# In[ ]:


import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt

TEXT_COL  = "Text_cleaned"
LABEL_COL = "Response_Label"

# Prepare data
mask = df[LABEL_COL].notna()
X_text = df.loc[mask, TEXT_COL].astype(str)
y_text = df.loc[mask, LABEL_COL].astype(str)

X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    X_text, y_text, test_size=0.2, random_state=42, stratify=y_text
)

# Vectorize → dense float32
tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
X_train = tfidf.fit_transform(X_train_text)

X_test  = tfidf.transform(X_test_text)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train_text)
y_test  = le.transform(y_test_text)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Metrics back in original label space
y_pred_labels = le.inverse_transform(y_pred)
print("Accuracy:", f"{accuracy_score(y_test_text, y_pred_labels):.4f}")
print(classification_report(y_test_text, y_pred_labels))

# Confusion matrix (normalized)
all_labels = sorted(le.classes_)
cm = confusion_matrix(y_test_text, y_pred_labels, labels=all_labels)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(10,8))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=all_labels, yticklabels=all_labels)
plt.title("Random Forest — Confusion Matrix (Normalized)")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.show()


# # Decision Tree Classifier

# In[ ]:


# Create a mask for non-null response labels
mask = df['Response_Label'].notna()

# Check if we need to process the text data
if 'X_train' not in locals() or 'X_test' not in locals():
    X_text = df.loc[mask, 'Text_cleaned'].astype(str)
    y_text = df.loc[mask, 'Response_Label'].astype(str)

    # Split data
    X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
        X_text, y_text, test_size=0.2, random_state=42, stratify=y_text
    )

    # Vectorize TF-IDF
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    X_train = tfidf.fit_transform(X_train_text).toarray()
    X_test  = tfidf.transform(X_test_text).toarray()

    # Encode labels
    y_train = le.fit_transform(y_train_text)
    y_test = le.transform(y_test_text)
else:
    # If X_train, X_test, y_train, y_test are already defined (e.g., from SentenceTransformer)
    # Ensure y_train and y_test are encoded using the same label encoder
    all_labels = np.concatenate([y_train, y_test])
    le.fit(all_labels)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)


# Define hyperparameter grid
params = {
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}

# Initialize base model
dt = DecisionTreeClassifier(random_state=42)

# Perform grid search with 3-fold cross-validation
grid = GridSearchCV(dt, param_grid=params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

# Best model
best_dt_model = grid.best_estimator_

# Predict using the best model
y_pred_dt = best_dt_model.predict(X_test)

# Evaluate
print("Best Hyperparameters:", grid.best_params_)
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")

# Visualize confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt, xticks_rotation=45, display_labels=le.classes_)
plt.title("Decision Tree - Confusion Matrix")
plt.show()


# # Testing model for Overfitting

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Split data 
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['Text_cleaned'], 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# TF-IDF: fit only on the training text
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# # Baseline check

# In[ ]:


import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

dum = DummyClassifier(strategy="most_frequent")
dum.fit(X_train, y_train)
y_base = dum.predict(X_test)
print("Baseline (most frequent) — acc:", accuracy_score(y_test, y_base))
print("Baseline (most frequent) — macro F1:", f1_score(y_test, y_base, average="macro"))



# # Checking Decision Tree for Overfitting

# In[ ]:


from sklearn.metrics import accuracy_score, f1_score

# Training predictions
y_train_pred = best_dt_model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred, average="macro")

# Test predictions
y_test_pred = best_dt_model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average="macro")

print(f"Train Accuracy: {train_acc:.3f}, Train Macro-F1: {train_f1:.3f}")
print(f"Test Accuracy:  {test_acc:.3f}, Test Macro-F1:  {test_f1:.3f}")


# High Accuracy with both train and Test almost certainly points to Overfitting

# # Mean Cross Validation

# In[ ]:


X_train_text, X_test_text, y_train, y_test = train_test_split(df['Description_cleaned'], y, stratify=y, random_state=42)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(dt, X_train, y_train, cv=5, scoring="f1_macro")
print("CV Macro-F1:", scores.mean())


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=15, min_samples_leaf=10, random_state=42)


# 

# # MLPClassifier (Multi-Layer Perceptron) for Neural Network

# In[ ]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=100,
    random_state=42,
    verbose=True
)

mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)

# Evaluate
from sklearn.metrics import classification_report, accuracy_score
print("MLPClassifier Classification Report:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_mlp):.4f}")
print("\nClassification Report:\n", classification_report(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred_mlp)))

# Confusion Matrix Plot
ConfusionMatrixDisplay.from_predictions(
    label_encoder.inverse_transform(y_test),
    label_encoder.inverse_transform(y_pred_mlp),
    display_labels=label_encoder.classes_,
    xticks_rotation=45,
    cmap='Blues'
)
plt.title("MLP - Confusion Matrix")
plt.tight_layout()
plt.show()


# # XGBoost

# In[ ]:


import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# Step 1: Upsample minority classes
max_size = df['Response_Label'].value_counts().max()

df_upsampled = (
    df.groupby('Response_Label', group_keys=False)
    .apply(lambda x: resample(x, replace=True, n_samples=max_size, random_state=42))
    .reset_index(drop=True)
)

# Step 2: Encode text with SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X = embedder.encode(df_upsampled['Description_cleaned'].tolist(), show_progress_bar=True)

# Step 3: Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df_upsampled['Response_Label'])

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Step 5: Train XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred_xgb = xgb_model.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(y_pred_xgb)

print("Balanced class distribution:")
print(df_upsampled['Response_Label'].value_counts())
print("Accuracy: {accuracy_score(label_encoder.inverse_transform(y_test), y_pred_labels):.4f}")
print("Classification Report:\n", classification_report(
    label_encoder.inverse_transform(y_test), y_pred_labels
))


# In[ ]:


import matplotlib.pyplot as plt
import xgboost as xgb

xgb.plot_tree(xgb_model, num_trees=1, rankdir='LR')
plt.rcParams['figure.figsize'] = [500, 250]
plt.rcParams['figure.dpi'] = 400
plt.title("XGBoost - First Tree")
plt.tight_layout()
plt.show()


# # **6. Narrative Conversion**

# # Prediction - Logic Loop

# In[ ]:


def get_best_cosine_match(input_text, reference_df, tfidf_vectorizer):
    input_vec = tfidf_vectorizer.transform([input_text])
    ref_vecs = tfidf_vectorizer.transform(reference_df['Description_cleaned'])
    similarities = cosine_similarity(input_vec, ref_vecs).flatten()

    best_idx = similarities.argmax()
    best_text = reference_df.iloc[best_idx]['Text_cleaned']
    best_score = similarities[best_idx]

    return best_text, best_score


# # Narrative Conversion

# In[ ]:


import os
client = OpenAI(api_key="",)
from groq import Groq

# Sample data to convert into narrative
data_dict = {
    'data': df[['Description_cleaned', 'Response_Label']].sample(5).to_string(index=False)
}


# OpenAI API Key
client  = Groq(
    api_key=("GET_GROQ_API"),
)

# Generate narrative summary using a language model
chat_completion = client.chat.completions.create(
    messages=[
        {"role": "user", "content": f"""You are a smart diagnostic assistant.
Analyze the following maintenance data and return a readable summary of what’s happening, what actions were taken, and what patterns you notice. Be brief, insightful, and professional.

Maintenance Data:
{data_dict['data']}
"""}
    ],
    model="llama3-8b-8192",
)

narrative = chat_completion.choices[0].message.content
print(narrative.replace('\n', ' '))


# In[ ]:


import joblib
from pathlib import Path

# Define the missing functions and model pipeline
def _llm_narrative(prompt):
    # This is a simple placeholder function - replace with actual LLM implementation
    return """Based on the symptoms, this appears to be a mechanical issue with the conveyor belt system. 
The belt squeal and hot motor near the gearbox suggest belt slippage or misalignment. 
First, check if the belt is properly tensioned and aligned to prevent further damage. 
Inspect the gearbox for proper lubrication and signs of wear. 
Plan to clean and lubricate the system, adjust belt tension, and if problems persist, consider replacing worn components in the drive system."""

def retrieve_neighbors(description, top_label, k=5, restrict=True):
    # Placeholder function to simulate retrieving similar cases
    return [
        {"label": "Belt Adjustment", "desc": "Conveyor belt slipping and making noise", "notes": "Adjusted tension and realigned belt", "score": 0.92},
        {"label": "Gearbox Repair", "desc": "Motor running hot with grinding noise", "notes": "Replaced worn gears and added lubricant", "score": 0.85},
        {"label": "Motor Cooling", "desc": "Overheating motor causing trips", "notes": "Cleaned vents and improved airflow", "score": 0.78}
    ]

# Define a function to get top actions 
def get_top_actions(description, k=3):
    # Placeholder function to simulate model predictions
    return [
        ("Belt Adjustment", 0.75),
        ("Gearbox Inspection", 0.65),
        ("Motor Cooling", 0.45)
    ]

def generate_narrative(description, topk_actions=3, k_neighbors=5, restrict_neighbors=True):
    # Get top actions if not provided
    if isinstance(topk_actions, int):
        top_actions = get_top_actions(description, k=topk_actions)
    else:
        top_actions = topk_actions

    # Get top label for neighbor retrieval
    top_label = top_actions[0][0] if top_actions else ""

    # Nearest historical cases
    neighbors = retrieve_neighbors(description, top_label, k=k_neighbors, restrict=restrict_neighbors)

    # Build a compact, model-agnostic prompt
    examples = []
    for n in neighbors:
        ex = f"- Label: {n.get('label','')}\n  Desc: {n.get('desc','')[:200]}\n  Notes: {n.get('notes','')[:220]}"
        examples.append(ex)
    examples_text = "\n".join(examples) if examples else "No close historical cases were retrieved."

    preds_text = "\n".join([f"{i+1}. {lbl}  ({prob:.1%})" for i,(lbl,prob) in enumerate(top_actions)])

    prompt = f"""You are a maintenance triage helper. 
Given a short problem description, provide a brief, practical narrative that suggests next steps.
Be concise and professional—no fluff. Avoid stating you are an AI.

Problem Description:
{description}

Top candidate actions (model):
{preds_text}

Nearest historical cases:
{examples_text}

Write 3–6 sentences covering:
1) What's most likely going on.
2) What to check first (quick wins / safety).
3) A concrete action plan.
4) If uncertain, suggest 1–2 alternate paths.
"""

    narrative = _llm_narrative(prompt)
    return {
        "description": description,
        "top_actions": top_actions,
        "neighbors": neighbors,
        "narrative": narrative
    }

# ---------- 5) Example usage ----------
example_text = "Conveyor stopped, belt squeal, motor running hot near gearbox; intermittent trip on startup."
out = generate_narrative(example_text, topk_actions=3, k_neighbors=5, restrict_neighbors=True)

print("— Problem —")
print(out["description"])
print("\n— Top actions —")
for a,p in out["top_actions"]:
    print(f"  • {a}: {p:.1%}")
print("\n— Narrative —")
print(out["narrative"])
print("\n— Similar cases —")
for i, n in enumerate(out["neighbors"], 1):
    print(f"[{i}] ({n.get('label','')}, sim={n.get('score',0):.2f})")
    print("    Desc :", (n.get('desc','') or "")[:140])
    print("    Notes:", (n.get('notes','') or "")[:140])


# In[ ]:




