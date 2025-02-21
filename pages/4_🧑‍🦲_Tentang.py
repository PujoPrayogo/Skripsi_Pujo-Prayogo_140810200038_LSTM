import streamlit as st
from streamlit_extras.badges import badge
from streamlit_extras.let_it_rain import rain 

st.set_page_config(
    page_title="About CuacaJakpus",
    page_icon = "⛈️",
)
st.title("🧑‍🦲Tentang Aplikasi")

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://pluviophile.net/wp-content/uploads/cloudy-weather-wallpaper.jpg");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}
[data-testid="stHeader"] {
background: rgba(0,0,0,0);
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

def emoji_rain():
    rain(
        emoji="💧",
        font_size=8,
        falling_speed=5,
        animation_length="infinite",
    )
emoji_rain()

with st.sidebar:
    st.sidebar.title("⛈️CuacaJakpus")
    badge(type="github", name="PujoPrayogo/Skripsi_Pujo-Prayogo_140810200038_LSTM")
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/id/8/80/Lambang_Universitas_Padjadjaran.svg")
    st.write("©Pujo 2025")

st.image("https://perikanan.psdku.unpad.ac.id/wp-content/uploads/2019/12/unpad-logo.png", caption="")
col1, col2= st.columns(2)


col1.markdown("#### Nama     : Pujo Prayogo")
col1.markdown("#### NPM      : 140810200038")
col1.markdown("#### Angkatan : 2020")

col2.write("Aplikasi ini adalah aplikasi untuk memprediksi data parameter cuaca menggunakan model LSTM.")
col2.write("Aplikasi ini dibuat untuk memenuhi kebutuhan Tugas Akhir Prodi S1-Teknik Informatika Universitas Padjadajaran pada tahun ajaran genap 2024/2025.")