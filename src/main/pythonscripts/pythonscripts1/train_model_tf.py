import pandas as pd
import pickle
import os
import re
import nltk
import numpy as np

from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack

nltk.download('stopwords')

# === RUTAS ===
TRUE_EN = "src/main/resources/data/True.csv"
FAKE_EN = "src/main/resources/data/Fake.csv"
TRUE_ES = "src/main/resources/data/onlytrue1000.csv"
FAKE_ES = "src/main/resources/data/onlyfakes1000.csv"

MODEL_GRID_PATH = "src/main/resources/data/model_grid.pkl"
VECTORIZER_PATH = "src/main/resources/data/vectorizer.pkl"

# === FUNCIONES ===
def load_and_standardize(path, label):
    df = pd.read_csv(path)
    df.columns = [col.lower() for col in df.columns]
    if 'text' not in df.columns:
        raise ValueError(f"❌ ERROR: {path} no tiene columna 'text'. Columnas encontradas: {df.columns.tolist()}")
    if 'title' not in df.columns:
        df['title'] = ""
    df = df[['title', 'text']].copy()
    df['label'] = label
    return df

def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.lower()

def extract_features(text):
    features = {}
    features['length'] = len(text)
    features['exclamation'] = text.count('!')
    features['question'] = text.count('?')
    features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    return list(features.values())

# === CARGA Y PREPROCESAMIENTO ===
dfs = [
    load_and_standardize(TRUE_EN, 1),
    load_and_standardize(FAKE_EN, 0),
    load_and_standardize(TRUE_ES, 1),
    load_and_standardize(FAKE_ES, 0)
]
data = pd.concat(dfs, ignore_index=True)
data['text'] = (data['title'] + " " + data['text']).apply(clean_text)

texts = data['text'].tolist()
labels = data['label'].tolist()

# === VECTORIZACIÓN GLOBAL ===
vectorizer = TfidfVectorizer(
    stop_words=stopwords.words('english') + stopwords.words('spanish'),
    max_features=15000,
    ngram_range=(1, 3),
    sublinear_tf=True
)
vectorizer.fit(texts)

# === VALIDACIÓN CRUZADA ESTRATIFICADA ===
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
accuracies = []

for train_idx, val_idx in kfold.split(texts, labels):
    print(f"\n📂 Training fold {fold}/5")
    fold += 1

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    X_train_vec = vectorizer.transform(train_texts)
    X_val_vec = vectorizer.transform(val_texts)

    X_train_feats = [extract_features(txt) for txt in train_texts]
    X_val_feats = [extract_features(txt) for txt in val_texts]

    X_train_final = hstack([X_train_vec, X_train_feats])
    X_val_final = hstack([X_val_vec, X_val_feats])

    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear', 'saga']
    }

    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train_final, train_labels)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_val_final)
    acc = accuracy_score(val_labels, y_pred)
    accuracies.append(acc)

    print(f"✅ Fold Accuracy: {acc * 100:.2f}%")
    print("🧠 Best Params:", grid_search.best_params_)
    print("📊 Report:")
    print(classification_report(val_labels, y_pred))

# === PROMEDIO DE DESEMPEÑO GENERAL ===
mean_acc = np.mean(accuracies)
print(f"\n🎯 Mean Cross-Validated Accuracy: {mean_acc * 100:.2f}%")

# === ENTRENAMIENTO FINAL Y GUARDADO ===
print("\n📦 Entrenando modelo final con todo el conjunto de datos...")
X_all_vec = vectorizer.transform(texts)
X_all_feats = [extract_features(txt) for txt in texts]
X_all_final = hstack([X_all_vec, X_all_feats])

final_model = LogisticRegression(C=1, penalty='l2', solver='liblinear', max_iter=1000)
final_model.fit(X_all_final, labels)

os.makedirs(os.path.dirname(MODEL_GRID_PATH), exist_ok=True)
with open(MODEL_GRID_PATH, 'wb') as f:
    pickle.dump(final_model, f)
with open(VECTORIZER_PATH, 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Modelo final guardado con validación cruzada.")