# ==============================================
#   PHISHING URL DETECTOR (ML Model)
#   AI LAB PROJECT
#   Dataset: Kaggle Phishing URL Dataset
# ==============================================

import pandas as pd
import numpy as np
import re
import urllib.parse
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix
)

# =============================================
#   STEP 1: FEATURE EXTRACTION FUNCTIONS
# =============================================

def get_url_length(url):
    """Long URLs are often used to hide suspicious parts."""
    return 1 if len(url) > 54 else 0

def has_at_symbol(url):
    """@ symbol tricks browser — everything before @ is ignored."""
    return 1 if "@" in url else 0

def has_ip_address(url):
    """Phishing sites use IP instead of domain name."""
    pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    return 1 if re.search(pattern, url) else 0

def has_https(url):
    """Legitimate sites almost always use HTTPS."""
    return 1 if url.lower().startswith("https") else 0

def has_suspicious_words(url):
    """Common phishing keywords found in URL."""
    bad_words = [
        'verify', 'update', 'login', 'secure', 'account',
        'banking', 'confirm', 'free', 'lucky', 'winner',
        'password', 'signin', 'webscr', 'ebayisapi', 'paypal'
    ]
    url_lower = url.lower()
    count = sum(1 for word in bad_words if word in url_lower)
    return 1 if count >= 2 else 0

def count_subdomains(url):
    """Too many subdomains = suspicious."""
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        return 1 if domain.count('.') > 3 else 0
    except:
        return 0

def count_dashes(url):
    """Multiple dashes in domain are a red flag."""
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        return 1 if domain.count('-') > 1 else 0
    except:
        return 0

def get_url_depth(url):
    """Deep URL paths can hide malicious endpoints."""
    try:
        parsed = urllib.parse.urlparse(url)
        return 1 if parsed.path.count('/') > 4 else 0
    except:
        return 0

def has_double_slash(url):
    """Double slash after http:// indicates redirection trick."""
    return 1 if '//' in url[7:] else 0

def get_domain_length(url):
    """Unusually long domain names are suspicious."""
    try:
        parsed = urllib.parse.urlparse(url)
        return 1 if len(parsed.netloc) > 20 else 0
    except:
        return 0

def extract_features(url):
    """Extract all 10 features from a single URL."""
    return [
        get_url_length(url),       # Feature 1
        has_at_symbol(url),        # Feature 2
        has_ip_address(url),       # Feature 3
        has_https(url),            # Feature 4
        has_suspicious_words(url), # Feature 5
        count_subdomains(url),     # Feature 6
        count_dashes(url),         # Feature 7
        get_url_depth(url),        # Feature 8
        has_double_slash(url),     # Feature 9
        get_domain_length(url)     # Feature 10
    ]

FEATURE_NAMES = [
    'url_length', 'has_at_symbol', 'has_ip_address',
    'has_https', 'suspicious_words', 'subdomain_count',
    'dash_count', 'url_depth', 'double_slash', 'domain_length'
]

# ============================================================
#   STEP 2: LOAD DATASET
# ============================================================

print("=" * 55)
print("   PHISHING URL DETECTOR — ML Model Training")
print("=" * 55)


DATASET_FILE = "dataset.csv"   

try:
    df = pd.read_csv(DATASET_FILE)
    print(f"\n Dataset loaded successfully!")
    print(f"   Total URLs : {len(df)}")
    print(f"   Columns    : {list(df.columns)}")
except FileNotFoundError:
    print(f"\n ERROR: '{DATASET_FILE}' not found!")
    print("   Please download the dataset from Kaggle and place")
    print(f"  it in the same folder as this script.")
    exit()

# -------------------------------------------
# Auto-detect URL column and label column
# -------------------------------------------
url_col   = None
label_col = None

for col in df.columns:
    if 'url' in col.lower():
        url_col = col
    if 'label' in col.lower() or 'status' in col.lower() or 'result' in col.lower():
        label_col = col

if not url_col or not label_col:
    print("\n Could not find URL or Label column automatically.")
    print(f"  Columns available: {list(df.columns)}")
    print("   Please set url_col and label_col manually below.")
    exit()

print(f"\n   URL Column   : '{url_col}'")
print(f"   Label Column : '{label_col}'")

# Show class distribution
print(f"\n Dataset Distribution:")
print(df[label_col].value_counts().to_string())

# ====================================
#   STEP 3: FEATURE EXTRACTION
# ====================================

print("\n Extracting features from URLs...")
print("   (This may take a minute for large datasets...)")

df = df.dropna(subset=[url_col, label_col])
df[url_col] = df[url_col].astype(str)

X = np.array(df[url_col].apply(extract_features).tolist())
y = np.array(df[label_col])

# If labels are strings like 'phishing'/'legitimate', convert to 0/1
if y.dtype == object:
    unique_vals = list(set(y))
    print(f"\n   Label values found: {unique_vals}")

    # Map phishing-related words to 1, rest to 0

    phishing_words = ['phishing', 'phish', 'bad', 'malicious', '1']
    y = np.array([1 if str(v).lower() in phishing_words else 0 for v in y])

print(f" Features extracted!")
print(f" Shape: {X.shape}  ({X.shape[0]} URLs × {X.shape[1]} features)")

# =================================
#   STEP 4: TRAIN / TEST SPLIT
# =================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n Data Split:")
print(f"   Training Set : {len(X_train)} URLs (80%)")
print(f"   Testing Set  : {len(X_test)} URLs  (20%)")

# ======================================
#   STEP 5: TRAIN MODELS & COMPARE
# ======================================

print("\n Training ML Models...")
print("-" * 45)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

results    = {}
trained    = {}

for name, model in models.items():
    print(f"\n   Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    results[name] = {
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "F1 Score":  f1
    }
    trained[name] = model
    print(f"  Done — Accuracy: {acc*100:.2f}%")

# ==================================
#   STEP 6: EVALUATION REPORT
# ==================================

print("\n" + "=" * 55)
print("   MODEL EVALUATION RESULTS")
print("=" * 55)

for name, metrics in results.items():
    print(f"\n {name}")
    print(f"   Accuracy  : {metrics['Accuracy']*100:.2f}%")
    print(f"   Precision : {metrics['Precision']*100:.2f}%")
    print(f"   Recall    : {metrics['Recall']*100:.2f}%")
    print(f"   F1 Score  : {metrics['F1 Score']*100:.2f}%")

# Pick the best model
best_name  = max(results, key=lambda k: results[k]["F1 Score"])
best_model = trained[best_name]

print(f"\n Best Model: {best_name}")
print(f"   F1 Score  : {results[best_name]['F1 Score']*100:.2f}%")

# Detailed classification report for best model
y_pred_best = best_model.predict(X_test)
print(f"\n Detailed Report ({best_name}):")
print(classification_report(y_test, y_pred_best,
      target_names=['Safe (0)', 'Phishing (1)']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
print(" Confusion Matrix:")
print(f"   True Safe    (correctly safe)     : {cm[0][0]}")
print(f"   False Alarm  (safe called phish)  : {cm[0][1]}")
print(f"   Missed       (phish called safe)  : {cm[1][0]}")
print(f"   True Phishing (correctly caught)  : {cm[1][1]}")

# Feature importance (Random Forest only)
if best_name == "Random Forest":
    print(f"\n Feature Importance ({best_name}):")
    importances = best_model.feature_importances_
    feat_imp    = sorted(zip(FEATURE_NAMES, importances),
                         key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp:
        bar = "█" * int(imp * 100)
        print(f"   {feat:<22} {imp:.4f}  {bar}")

# ===================================
#   STEP 7: SAVE THE BEST MODEL
# ===================================

MODEL_FILE = "phishing_model.pkl"
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(best_model, f)

print(f"\n Model saved as '{MODEL_FILE}'")

# =========================================================
#   STEP 8: PREDICTION FUNCTION (for testing new URLs)
# =========================================================

def predict_url(url, model=best_model):
    """
    Pass any URL and get prediction.
    Returns: result string + probability
    """
    features  = np.array(extract_features(url)).reshape(1, -1)
    pred      = model.predict(features)[0]
    prob      = model.predict_proba(features)[0]

    label     = "PHISHING" if pred == 1 else "SAFE"
    confidence = prob[1] * 100 if pred == 1 else prob[0] * 100

    return pred, label, confidence

# =======================================
#   STEP 9: TEST WITH SAMPLE URLs
# =======================================

print("\n" + "=" * 55)
print("   LIVE URL PREDICTIONS (Sample Test)")
print("=" * 55)

sample_urls = [
    "https://www.google.com",
    "https://github.com/login",
    "http://paypa1.com/login@verify-account.php",
    "http://192.168.1.1/bank/update/verify",
    "http://secure-paypal-login-verify.com/confirm?id=abc",
    "https://www.youtube.com",
    "http://free-winner-lucky-prize.com/claim?user=123",
]

print(f"\n{'URL':<45} {'Result':<15} {'Confidence'}")
print("-" * 75)
for url in sample_urls:
    pred, label, conf = predict_url(url)
    short_url = url[:44] + "…" if len(url) > 44 else url
    print(f"{short_url:<45} {label:<15} {conf:.1f}%")

# ============================================
#   INTERACTIVE MODE — Test your own URL
# ============================================

print("\n" + "=" * 55)
print("TEST YOUR OWN URL")
print("=" * 55)

while True:
    user_url = input("\nEnter URL to check (or 'quit' to exit): ").strip()
    if user_url.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye! 👋")
        break
    if not user_url:
        continue

    pred, label, conf = predict_url(user_url)
    features = extract_features(user_url)

    print(f"\nResult     : {label}")
    print(f"Confidence : {conf:.1f}%")
    print(f"\nFeature Breakdown:")
    for fname, fval in zip(FEATURE_NAMES, features):
        status = "🔴" if fval == 1 else "🟢"
        print(f"  {status} {fname:<22}: {fval}")
