import streamlit as st
import time
import json

# 1. Konfigurasi Halaman (Bikin UI Clean & Rapi)
st.set_page_config(
    page_title="Layanan Publik Pemda",
    page_icon="🏛️",
    layout="centered"
)

# Judul dan Deskripsi
st.title("🏛️ Portal Pengaduan AI Pemda")
st.markdown("Sampaikan laporan atau pertanyaan Anda. Sistem AI kami akan menganalisis maksud (intent) dan sentimen dari laporan Anda secara otomatis.")
st.divider()

# 2. Kolom Input Teks
user_input = st.text_area(
    "📝 Masukkan Laporan/Pertanyaan Anda di sini:", 
    placeholder="Contoh: Jalan di depan pasar rusak parah dan sering banjir...", 
    height=130
)

# 3. Tombol Analisis
if st.button("🚀 Analisis Laporan", use_container_width=True):
    
    # Validasi input kosong
    if not user_input.strip():
        st.warning("⚠️ Laporan tidak boleh kosong, ya! Silakan ketik sesuatu.")
    else:
        # 4. Fitur Robustness (Try-Except Block)
        try:
            # 5. Loading Spinner
            with st.spinner('Mengubah teks ke Word2Vec & menganalisis intent...'):
                
                # Simulasi waktu loading backend (hapus ini nanti kalau digabung dengan backend asli)
                time.sleep(2)
                
                # Di sinilah nanti kamu memanggil fungsi backend dari temanmu
                # Contoh: hasil_backend = model.predict(user_input)

                # 6. Menampilkan Mock Respons JSON (Syarat SOW REST API)
                mock_response = {
                    "status_code": "200 OK",
                    "data": {
                        "input_asli": user_input,
                        "intent_terdeteksi": "Pengaduan Infrastruktur", 
                        "label_sentimen": "Negatif",
                        "confidence_score": 0.92
                    },
                    "message": "Berhasil memproses representasi Dense Word2Vec."
                }
                
                st.success("✅ Analisis Berhasil!")
                
                st.subheader("📊 Dokumentasi Output (Mock JSON)")
                st.json(mock_response)
                
        except Exception as e:
            # Penanganan error agar web tidak crash (Human-readable error)
            st.error("❌ Mohon maaf, terjadi kesalahan pada sistem kami.")
            st.info(f"Detail error untuk developer: {str(e)}")