import pandas as pd
import pickle
import os
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
from skopt import BayesSearchCV

nltk.download('stopwords')

# === RUTAS ===
TRUE_EN = "src/main/resources/data/True.csv"
FAKE_EN = "src/main/resources/data/Fake.csv"
TRUE_ES = "src/main/resources/data/onlytrue1000.csv"
FAKE_ES = "src/main/resources/data/onlyfakes1000.csv"

MODEL_PATH = "src/main/resources/data/model_xgb.pkl"
MODEL_GRID_PATH = "src/main/resources/data/model_xgb_grid.pkl"
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
    text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.lower().strip()

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
X = data['text']
y = data['label']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === VECTORIZACIÓN MEJORADA ===
vectorizer = TfidfVectorizer(
    stop_words=stopwords.words('english') + stopwords.words('spanish'),
    max_features=15000,
    ngram_range=(1, 3),
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train_raw)
X_test_vec = vectorizer.transform(X_test_raw)

X_train_feats = [extract_features(txt) for txt in X_train_raw]
X_test_feats = [extract_features(txt) for txt in X_test_raw]

X_train_final = hstack([X_train_vec, X_train_feats])
X_test_final = hstack([X_test_vec, X_test_feats])

# === MODELO BASE XGBOOST ===
print("\nEntrenando modelo base (XGBoost)...")
base_model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)
base_model.fit(X_train_final, y_train)
y_pred_base = base_model.predict(X_test_final)
print(f"[BASE] Accuracy: {accuracy_score(y_test, y_pred_base)*100:.2f}%")

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(base_model, f)
with open(VECTORIZER_PATH, 'wb') as f:
    pickle.dump(vectorizer, f)
print("✅ Modelo base y vectorizador guardados.")

# === BAYESIAN OPTIMIZATION ===
print("\n🔍 Iniciando búsqueda bayesiana de hiperparámetros...")
param_grid = {
    'n_estimators': (100, 300),
    'max_depth': (4, 8),
    'learning_rate': (0.01, 0.1, 'log-uniform'),
    'subsample': (0.6, 1.0, 'uniform'),
    'colsample_bytree': (0.6, 1.0, 'uniform'),
    'gamma': (0.0, 0.2)
}

search = BayesSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    search_spaces=param_grid,
    n_iter=30,
    cv=3,
    n_jobs=-1,
    verbose=2
)

search.fit(X_train_final, y_train)
best_model = search.best_estimator_

y_pred_grid = best_model.predict(X_test_final)
print(f"[BAYES SEARCH] Accuracy: {accuracy_score(y_test, y_pred_grid)*100:.2f}%")
print("🧠 Mejores parámetros:", search.best_params_)
print("📄 Reporte de clasificación:")
print(classification_report(y_test, y_pred_grid))

with open(MODEL_GRID_PATH, 'wb') as f:
    pickle.dump(best_model, f)
print("✅ Modelo optimizado guardado como model_xgb_grid.pkl")

#Se reemplazó GridSearchCV por BayesSearchCV para optimización más eficiente.
#•	Se amplió el rango de búsqueda de hiperparámetros.
#•	Se mantienen las características manuales (length, exclamation, etc.).
#•	El vectorizador usa trigramas, sublinear_tf, y stopwords en español e inglés.