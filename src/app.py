import os
import json
import pickle
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="E-Pemda Smart Chatbot", page_icon="🏛️", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 2rem; }
    
    div.stButton > button:first-child {
        background-color: #1E3A8A !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #3B82F6 !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .response-card {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .response-text { color: #065F46; font-size: 1.05rem; white-space: pre-line; }
    
    div[data-testid="stSidebar"] div.stButton > button {
        text-align: left !important;
        display: block !important;
        width: 100% !important;
        padding: 0.6rem 1rem !important;
        line-height: 1.3 !important;
    }
    
    div[data-testid="stSidebar"] .stButton:last-child > button {
        background-color: #DC2626 !important;
        color: white !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        text-align: center !important;
        margin-top: 2rem !important;
    }
    div[data-testid="stSidebar"] .stButton:last-child > button:hover {
        background-color: #B91C1C !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(BASE_DIR, "models")
dataset_path = os.path.join(BASE_DIR, "data", "knowledge_base.json")

@st.cache_resource
def load_pipeline_assets():
    try:
        with open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(models_dir, "dense_embeddings.pkl"), "rb") as f:
            dense_embeddings = pickle.load(f)
        with open(os.path.join(models_dir, "intent_mapping.pkl"), "rb") as f:
            intent_mapping = pickle.load(f)
        with open(dataset_path, "r", encoding="utf-8") as f:
            knowledge_base = json.load(f)
        return vectorizer, dense_embeddings, intent_mapping, knowledge_base
    except Exception as e:
        st.error(f"❌ System Failure: {e}")
        st.stop()

vectorizer, dense_embeddings, intent_mapping, knowledge_base = load_pipeline_assets()

def preprocess_text(text):
    return str(text).lower().strip()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_view' not in st.session_state:
    st.session_state.current_view = None

with st.sidebar:
    st.markdown("### 📜 Riwayat Konsultasi")
    
    if not st.session_state.chat_history:
        st.caption("Belum ada riwayat pertanyaan.")
    else:
        for idx, chat in enumerate(reversed(st.session_state.chat_history)):
            actual_idx = len(st.session_state.chat_history) - 1 - idx
            short_query = chat['query'][:28] + "..." if len(chat['query']) > 28 else chat['query']
            button_label = f"{short_query}\n🕒 {chat['timestamp']}"
            
            if st.button(button_label, key=f"hist_{actual_idx}", use_container_width=True):
                st.session_state.current_view = st.session_state.chat_history[actual_idx]
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🗑️ Hapus Semua Riwayat", key="btn_clear_history", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_view = None
            st.rerun()

st.markdown('<div class="main-header">🏛️ Hub Layanan Publik Digital Pemda</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Asisten Pintar Berbasis Kecerdasan Buatan (AI) Kelompok AI03</div>', unsafe_allow_html=True)

tab_chat, tab_api = st.tabs(["💬 Konsultasi Warga (Chatbot)", "📄 Portal Pengembang (JSON REST API)"])

with tab_chat:
    st.write("### 🏢 Layanan Pengaduan & Prosedur Informasi")
    
    user_query = st.text_input("Silakan ketik pertanyaan Anda (Contoh: KTP saya patah cara gantinya gimana?):", 
                               placeholder="Tulis di sini...", key="main_input")
    
    if st.button("Tanyakan ke Asisten AI", key="btn_submit"):
        if not user_query.strip():
            st.warning("⚠️ Silakan ketik pertanyaan Anda terlebih dahulu.")
        else:
            with st.spinner("🤖 Mengkalkulasi kedekatan semantik teks..."):
                try:
                    clean_input = preprocess_text(user_query)
                    input_vector = vectorizer.transform([clean_input]).toarray()
                    
                    similarities = cosine_similarity(input_vector, dense_embeddings)[0]
                    best_match_idx = np.argmax(similarities)
                    confidence_score = similarities[best_match_idx]
                    predicted_intent = intent_mapping[best_match_idx]
                    
                    if confidence_score < 0.15:
                        predicted_intent = "tidak_dikenali"
                        bot_response = "😅 Maaf, saya belum dapat memahami konteks pertanyaan Anda secara spesifik. Layanan chatbot saat ini hanya berfokus pada informasi seputar prosedur pengurusan KTP, Kartu Keluarga (KK), Akta Kelahiran, KIA, dan Surat Pindah Domisili."
                    else:
                        bot_response = "Maaf, panduan aturan tidak ditemukan di basis pengetahuan."
                        for item in knowledge_base:
                            if item["intent"] == predicted_intent:
                                bot_response = item["response"]
                                break
                    
                    new_chat_record = {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "query": user_query,
                        "normalized": clean_input,
                        "intent": predicted_intent,
                        "score": confidence_score,
                        "response": bot_response
                    }
                    
                    st.session_state.chat_history.append(new_chat_record)
                    st.session_state.current_view = new_chat_record
                    st.rerun()
                    
                except Exception as err:
                    st.error(f"❌ Terjadi kesalahan internal: {err}")

    if st.session_state.current_view:
        active_data = st.session_state.current_view
        st.write("---")
        st.write(f"🔍 **Menampilkan Hasil Pertanyaan:** *\"{active_data['query']}\"*")
        
        st.markdown(f"""
            <div class="response-card">
                <div style="font-weight: 700; font-size: 1.15rem; color: #065F46; margin-bottom: 0.5rem;">🏛️ Jawaban Resmi Terverifikasi:</div>
                <div class="response-text">{active_data['response']}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.info(f"💡 **Analisis Kecerdasan Semantik:** Intent terdeteksi sebagai **[{active_data['intent'].upper()}]** dengan tingkat kedekatan ruang vektor sebesar **{active_data['score']:.2%}**.")

with tab_api:
    st.write("### 🔑 Spesifikasi Data JSON Payload (Dinamis & Terisolasi)")
    st.write("Gunakan data payload di bawah ini untuk integrasi sistem eksternal:")
    
    if st.session_state.current_view:
        active_data = st.session_state.current_view
        api_payload = {
            "status": 200,
            "message": "Success",
            "endpoint": "/api/v1/chatbot/inference",
            "developer_team": "Kelompok AI03",
            "data": {
                "user_query": active_data["query"],
                "normalized_query": active_data["normalized"],
                "inference": {
                    "predicted_intent": active_data["intent"],
                    "confidence_score": float(np.round(active_data["score"], 4))
                },
                "expert_system_output": {
                    "response_text": active_data["response"]
                }
            }
        }
        st.success("🔄 Menampilkan data payload aktual dari pertanyaan aktif di Tab 1:")
        st.json(api_payload)
    else:
        st.warning("💡 Belum ada data transaksi aktif. Silakan kirim pertanyaan pada Tab 'Konsultasi Warga' terlebih dahulu untuk menjana payload JSON dinamis.")
        with st.expander("Lihat Skema/Blueprint Cetak Biru Default JSON"):
            st.json({"status": 200, "message": "Template Ready", "data": "Silakan kirim pertanyaan untuk mengisi node data ini."})

st.write("---")
st.markdown("<center style='color:#9CA3AF; font-size:0.8rem;'>© 2026 Universitas Cakrawala - Fakultas Komputer - Program Studi Ilmu Komputer</center>", unsafe_allow_html=True)