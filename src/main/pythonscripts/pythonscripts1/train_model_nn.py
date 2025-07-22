#!/usr/bin/env python3


import os
import re
import unicodedata
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# Configuración de entorno
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Estructura mejorada para manejar múltiples datasets con diferentes estructuras
DATA_PATHS = [
    # Datasets originales
    {"path": "src/main/resources/data/True.csv", "label": 0, "language": "en", "text_col": "text", "title_col": "title"},
    {"path": "src/main/resources/data/Fake.csv", "label": 1, "language": "en", "text_col": "text", "title_col": "title"},
    {"path": "src/main/resources/data/onlytrue1000.csv", "label": 0, "language": "es", "text_col": "content", "title_col": "title"},
    {"path": "src/main/resources/data/onlyfakes1000.csv", "label": 1, "language": "es", "text_col": "content", "title_col": "title"},

    # Nuevos datasets añadidos
    {"path": "src/main/resources/data/gossipcop_fake.csv", "label": 1, "language": "en", "text_col": "text", "title_col": "title"},
    {"path": "src/main/resources/data/gossipcop_real.csv", "label": 0, "language": "en", "text_col": "text", "title_col": "title"},
    {"path": "src/main/resources/data/politifact_fake.csv", "label": 1, "language": "en", "text_col": "text", "title_col": "title"},
    {"path": "src/main/resources/data/politifact_real.csv", "label": 0, "language": "en", "text_col": "text", "title_col": "title"}
]
MODEL_DIR = "src/main/resources/data/model_nn"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 10

def clean_text(t: str) -> str:
    t = unicodedata.normalize('NFKC', str(t))
    t = re.sub(r"http\S+|www\S+|https\S+", "", t, flags=re.MULTILINE)
    t = re.sub(r"@\w+|#\w+", "", t)
    t = re.sub(r'[^\w\s.,;:¿?¡!áéíóúñÁÉÍÓÚÑ]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t[:5000]

def load_and_preprocess():
    datasets = []
    for dataset_info in DATA_PATHS:
        try:
            df = pd.read_csv(dataset_info["path"])
            df.columns = [c.lower() for c in df.columns]

            # Manejo flexible de columnas
            text_col = dataset_info.get("text_col")
            title_col = dataset_info.get("title_col")

            if not text_col or text_col not in df.columns:
                # Intento automático de encontrar columna de texto
                text_col = next((c for c in ("text", "content", "body", "article") if c in df.columns), None)

            if not title_col or title_col not in df.columns:
                # Intento automático de encontrar columna de título
                title_col = "title" if "title" in df.columns else None

            if text_col:
                text = df[text_col].fillna("")
                title = df[title_col].fillna("") if title_col else ""
                df["full_text"] = (title + " " + text).apply(clean_text)
                df["label"] = dataset_info["label"]

                # Añadir metadata para análisis posterior
                df["source"] = os.path.basename(dataset_info["path"])
                df["language"] = dataset_info.get("language", "unknown")

                datasets.append(df[["full_text", "label", "source", "language"]])
                print(f"✅ Cargado {dataset_info['path']} - Muestras: {len(df)}")
            else:
                print(f"⚠️ No se encontró columna de texto en {dataset_info['path']}")

        except Exception as e:
            print(f"⚠️ Error cargando {dataset_info['path']}: {e}")

    if not datasets:
        raise RuntimeError("❌ No se cargó ningún dataset")

    data = pd.concat(datasets, ignore_index=True)

    # Análisis de distribución
    print("\n📊 Distribución por fuente:")
    print(data["source"].value_counts())

    print("\n🌍 Distribución por idioma:")
    print(data["language"].value_counts())

    print("\n🏷️ Distribución original de etiquetas:")
    print(data["label"].value_counts())

    # Balanceo de datos
    min_samples = min(data["label"].value_counts())
    balanced_data = pd.concat([
        data[data["label"] == 0].sample(min_samples, random_state=SEED),
        data[data["label"] == 1].sample(min_samples, random_state=SEED)
    ])

    print(f"\n⚖️ Datos balanceados - Muestras por clase: {min_samples}")
    return balanced_data.sample(frac=1, random_state=SEED)

print("📥 Cargando y balanceando datos...")
data = load_and_preprocess()
print("\n📊 Distribución final de clases:\n", data["label"].value_counts())

train_df, test_df = train_test_split(
    data, test_size=0.2, stratify=data["label"], random_state=SEED
)
val_df, test_df = train_test_split(
    test_df, test_size=0.5, stratify=test_df["label"], random_state=SEED
)

print(f"📊 Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def create_dataset(df):
    texts = df["full_text"].tolist()
    labels = df["label"].values
    encodings = tokenizer(
        texts,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="tf"
    )
    return tf.data.Dataset.from_tensor_slices((
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"]
        },
        labels
    )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_dataset = create_dataset(train_df)
val_dataset = create_dataset(val_df)
test_dataset = create_dataset(test_df)

print("🧠 Construyendo modelo...")
model = TFAutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=2
)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

callbacks = [
    EarlyStopping(patience=3, monitor="val_loss", restore_best_weights=True),
    ModelCheckpoint(MODEL_DIR, save_best_only=True, monitor="val_accuracy"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2)
]

print("🚀 Entrenando modelo...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\n📊 Evaluando modelo...")
test_loss, test_acc = model.evaluate(test_dataset)
print(f"\n🧪 Precisión en test: {test_acc:.4f}")

y_true = test_df["label"].values
y_pred = np.argmax(model.predict(test_dataset)["logits"], axis=1)

print("\n📝 Reporte de clasificación:")
print(classification_report(y_true, y_pred))
print("\n📊 Matriz de confusión:")
print(confusion_matrix(y_true, y_pred))

# ======== EXPORTACIÓN ROBUSTA ========
print("\n💾 Exportando modelo en formato robusto...")

# 1. Guardar modelo completo con save_pretrained (genera tf_model.h5 y config)
model.save_pretrained(MODEL_DIR)

# 2. Guardar tokenizer
tokenizer.save_pretrained(MODEL_DIR)

# 3. Verificar y crear saved_model si es necesario
if not os.path.exists(os.path.join(MODEL_DIR, "saved_model")):
    print("🔧 Creando carpeta saved_model adicional...")
    model.save(os.path.join(MODEL_DIR, "saved_model"), save_format="tf")

# 4. Verificar archivos generados
print("\n✅ Archivos generados en", MODEL_DIR)
for f in os.listdir(MODEL_DIR):
    print(f"- {f}{' (carpeta)' if os.path.isdir(os.path.join(MODEL_DIR, f)) else ''}")

print("\n🎉 ¡Modelo exportado correctamente con todos los formatos necesarios!")