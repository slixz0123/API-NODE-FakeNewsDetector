#!/usr/bin/env bash
set -e

# 1. Crear venv si no existe
if [ ! -d "venv" ]; then
  echo "🛠️  Creando entorno virtual…"
  python3 -m venv venv
fi

# 2. Activar venv
echo "⚙️  Activando entorno virtual…"
source venv/bin/activate

# 3. Actualizar pip, setuptools y wheel
echo "📦  Actualizando pip, setuptools y wheel…"
python3 -m pip install --upgrade pip setuptools wheel

# 4. Instalar todas las dependencias
echo "📦  Instalando requirements.txt…"
python3 -m pip install -r requirements.txt

# 5. Descargar stopwords de NLTK
echo "🔤  Descargando stopwords de NLTK…"
python3 - <<EOF
import nltk
nltk.download('stopwords')
EOF

# 6. Entrenar los modelos
echo "🤖  Entrenando Logistic Regression…"
python3 src/main/pythonscripts/pythonscripts1/train_model.py

echo "🤖  Entrenando XGBoost…"
python3 src/main/pythonscripts/pythonscripts1/train_model_v2.py

echo "🤖  Entrenando TensorFlow…"
python3 src/main/pythonscripts/pythonscripts1/train_model_tf.py

echo "✅  Setup completado. Para activarlo siempre: source venv/bin/activate"