import sys
import pickle
import os
from langdetect import detect
from deep_translator import GoogleTranslator

translation_cache = {}

def translate_if_needed(text):
    if text in translation_cache:
        return translation_cache[text]

    try:
        lang = detect(text)
        if lang != "en":
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            translation_cache[text] = translated
            return translated
        return text
    except Exception as e:
        print(f"[⚠️] Language error: {e}")
        return text

def predict(title, text):
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "../../resources/data/model_xgb.pkl")
    vectorizer_path = os.path.join(base_path, "../../resources/data/vectorizer.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found at: {vectorizer_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    title_en = translate_if_needed(title)
    text_en = translate_if_needed(text)
    input_text = title_en + " " + text_en
    input_tfidf = vectorizer.transform([input_text])

    proba = model.predict_proba(input_tfidf)[0][1]
    threshold = 0.6
    label = int(proba >= threshold)

    # Imprimir solo el resultado necesario
    print(f"{label} {proba}", flush=True)

if __name__ == "__main__":
    try:
        if len(sys.argv) < 3:
            raise ValueError("Usage: python fake_news_predictor_v2.py <title> <text>")

        title = sys.argv[1]
        text = sys.argv[2]
        predict(title, text)
    except Exception as e:
        print("ERROR:", str(e), flush=True)
        sys.exit(1)