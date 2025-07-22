import pandas as pd
import pickle
import os
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

# === RUTAS ===
# Ruta absoluta a la raíz del proyecto
ROOT = Path(__file__).resolve().parents[4]
print("✅ Ruta raíz detectada:", ROOT)
print("✅ Dataset existe:", (ROOT / "src/main/resources/data/True.csv").exists())

# Rutas de los datasets
TRUE_EN = ROOT / "src" / "main" / "resources" / "data" / "True.csv"
FAKE_EN = ROOT / "src" / "main" / "resources" / "data" / "Fake.csv"
TRUE_ES = ROOT / "src" / "main" / "resources" / "data" / "onlytrue1000.csv"
FAKE_ES = ROOT / "src" / "main" / "resources" / "data" / "onlyfakes1000.csv"

# Rutas de salida
MODEL_PATH = ROOT / "src" / "main" / "resources" / "data" / "model.pkl"
MODEL_GRID_PATH = ROOT / "src" / "main" / "resources" / "data" / "model_grid.pkl"
VECTORIZER_PATH = ROOT / "src" / "main" / "resources" / "data" / "vectorizer.pkl"

# === FUNCIÓN ROBUSTA DE CARGA ===
def load_and_standardize(path, label):
    df = pd.read_csv(str(path))

    # Normalizar nombres de columnas
    columns = [col.lower() for col in df.columns]
    col_map = {col: col.lower() for col in df.columns}
    df.rename(columns=col_map, inplace=True)

    if 'text' not in df.columns:
        raise ValueError(f"❌ ERROR: El archivo {path} no tiene columna 'text'. Columnas encontradas: {df.columns.tolist()}")

    # Si no tiene título, lo creamos vacío
    if 'title' not in df.columns:
        df['title'] = ""

    df = df[['title', 'text']].copy()
    df['label'] = label
    return df

# === CARGA DE TODOS LOS DATOS ===
df_true_en = load_and_standardize(TRUE_EN, 1)
df_fake_en = load_and_standardize(FAKE_EN, 0)
df_true_es = load_and_standardize(TRUE_ES, 1)
df_fake_es = load_and_standardize(FAKE_ES, 0)

# Combinamos
data = pd.concat([df_true_en, df_fake_en, df_true_es, df_fake_es], ignore_index=True)

# === LIMPIEZA DE TEXTO ===
def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.lower()

data['text'] = (data['title'] + " " + data['text']).apply(clean_text)
X = data['text']
y = data['label']

# === SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === VECTORIZACIÓN ===
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1, 2)
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# === MODELO BASE ===
print("\nEntrenando modelo base (Logistic Regression)...")
base_model = LogisticRegression(max_iter=1000)
base_model.fit(X_train_tfidf, y_train)
y_pred_base = base_model.predict(X_test_tfidf)
print(f"[BASE] Accuracy: {accuracy_score(y_test, y_pred_base)*100:.2f}%")

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(base_model, f)
with open(VECTORIZER_PATH, 'wb') as f:
    pickle.dump(vectorizer, f)
print("✅ Modelo base y vectorizador guardados.")

# === GRIDSEARCHCV ===
print("\nIniciando búsqueda con GridSearchCV...")
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
    verbose=2
)
grid_search.fit(X_train_tfidf, y_train)
best_model = grid_search.best_estimator_

# === EVALUACIÓN FINAL ===
y_pred_grid = best_model.predict(X_test_tfidf)
print(f"[GRIDSEARCH] Accuracy: {accuracy_score(y_test, y_pred_grid)*100:.2f}%")
print("🧠 Mejores parámetros:", grid_search.best_params_)
print("📄 Reporte de clasificación:")
print(classification_report(y_test, y_pred_grid))

with open(MODEL_GRID_PATH, 'wb') as f:
    pickle.dump(best_model, f)
print("✅ Modelo optimizado guardado como model_grid.pkl")
