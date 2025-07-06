import streamlit as st
from streamlit_extras.badges import badge
import theme

st.set_page_config(
    page_title="About CuacaJakpus",
    page_icon = "🌦️",
)
st.title("🧑‍🦲Tentang Aplikasi")

theme.apply_theme() # Tema Halaman (Theme.py)

st.image("https://perikanan.psdku.unpad.ac.id/wp-content/uploads/2019/12/unpad-logo.png", caption="")

col1, col2= st.columns(2)
col1.markdown("#### Nama     : Pujo Prayogo")
col1.markdown("#### NPM      : 140810200038")
col1.markdown("#### Angkatan : 2020")

col2.write("Aplikasi ini adalah aplikasi untuk memprediksi data parameter cuaca menggunakan model LSTM.")
col2.write("Aplikasi ini dibuat untuk memenuhi kebutuhan Tugas Akhir Prodi S1-Teknik Informatika Universitas Padjadajaran pada tahun ajaran genap 2024/2025.")