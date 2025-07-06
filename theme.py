import streamlit as st
from streamlit_extras.badges import badge

def apply_theme():
    # CSS Background Theme
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1727249372967-c70430ba6847?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-position: center -120px;
    background-repeat: no-repeat;
    background-attachment: local;
    }
    [data-testid="stHeader"] {
    background: rgba(0,0,0,0);
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    with st.sidebar:
        st.sidebar.title("🌦️CuacaJakpus")
        st.write("Aplikasi prediksi cuaca pada tahun 2025 menggunakan model LSTM")
        badge(type="github", name="PujoPrayogo/Skripsi_Pujo-Prayogo_140810200038_LSTM")
        st.sidebar.image("https://upload.wikimedia.org/wikipedia/id/8/80/Lambang_Universitas_Padjadjaran.svg", width=100)
        st.write("©Pujo Prayogo | 2025")