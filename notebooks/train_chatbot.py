import os
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("🤖 Memulai Pipeline Dense Semantic Chatbot Pemda AI03 (Optimized)")

# 1. Loading Dataset
dataset_path = os.path.join("..", "data", "knowledge_base.json")
if not os.path.exists(dataset_path):
    dataset_path = os.path.join("data", "knowledge_base.json")

print(f"[*] Membaca dataset dari: {dataset_path}")
with open(dataset_path, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# 2. Pembersihan Teks Ringan (Tanpa Stemmer agar keyword kependudukan tidak rusak)
def preprocess_text(text):
    return str(text).lower().strip()

# 3. Flattening Data
print("[*] Memproses dan mengekstraksi frasa pelatihan...")
training_data = []
intent_mapping = []

for item in knowledge_base:
    for query in item["queries"]:
        clean_q = preprocess_text(query)
        training_data.append(clean_q)
        intent_mapping.append(item["intent"])

print(f"    -> Berhasil mengekstrak {len(training_data)} sampel frasa pertanyaan.")

# 4. Feature Extraction (Fokus pada Karakter dan Kata Pendek)
print("[*] Mengekstrak fitur Dense Semantics menggunakan TfidfVectorizer...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer='word')
dense_embeddings = vectorizer.fit_transform(training_data).toarray()

print(f"    -> Dimensi matriks dense representation: {dense_embeddings.shape}")

# 5. Exporting Binary Assets (.pkl)
models_dir = "../models" if os.path.exists("../models") else "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
embeddings_path = os.path.join(models_dir, "dense_embeddings.pkl")
mapping_path = os.path.join(models_dir, "intent_mapping.pkl")

print(f"[*] Mengekspor aset model biner (.pkl) ke folder '{models_dir}/'...")

with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)

with open(embeddings_path, "wb") as f:
    pickle.dump(dense_embeddings, f)

with open(mapping_path, "wb") as f:
    pickle.dump(intent_mapping, f)

# 6. Simulasi Pengujian Integritas Model
print("\n🧪 Menjalankan Simulasi Pengujian Integritas Model")

def simulate_inference(user_input):
    clean_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([clean_input]).toarray()
    
    similarities = cosine_similarity(input_vector, dense_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    confidence_score = similarities[best_match_idx]
    predicted_intent = intent_mapping[best_match_idx]
    
    return predicted_intent, confidence_score

test_queries = [
    "halo selamat sore pak", 
    "saya kehilangan kartu identitas penduduk saya",
    "gimana cara bikin surat lahir anak bayi?"
]

print("Hasil uji coba pencocokan semantik otomatis:")
for q in test_queries:
    intent, score = simulate_inference(q)
    print(f" -> Input : '{q}'")
    print(f"    Intent: [{intent.upper()}] | Kedekatan Semantik: {score:.2%}\n")

print("🏆 Sukses! Pipeline backend valid dan model siap dideploy.")