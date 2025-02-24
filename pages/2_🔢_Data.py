import streamlit as st
import matplotlib.pyplot as plt
from streamlit_extras.let_it_rain import rain 

st.set_page_config(
    page_title="Data CuacaJakpus",
    page_icon = "⛈️",
)

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

from streamlit_extras.badges import badge
with st.sidebar:
    st.sidebar.title("⛈️CuacaJakpus")
    badge(type="github", name="PujoPrayogo/Skripsi_Pujo-Prayogo_140810200038_LSTM")
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/id/8/80/Lambang_Universitas_Padjadjaran.svg")
    st.write("©Pujo 2025")

st.title("🔢Data yang digunakan")
st.markdown('Sumber : [POWER NASA](https://power.larc.nasa.gov/data-access-viewer/)')
st.write(f'Range : {st.session_state['start_date'].year} - {st.session_state['end_date'].year}')
st.write(f'Koordinat : {st.session_state['lat']}, {st.session_state['long']}')

features_name = [
    "Kelembaban_Tanah",
    "Kecepatan_Angin",
    "Tekanan_Permukaan",
    "Presipitasi",
    "Kelembaban_Udara",
    "Temperatur"
]

# List metric yang digunakan pada tiap parameter (berurutan dengan 'features')
metrics = ['0/1', 'm/s', 'kPa', 'mm/hari', '%', 'C']

df = st.session_state["df_all"]

features = [
    "GWETROOT", 
    "WS10M", 
    "PS", 
    "PRECTOTCORR",
    "RH2M", 
    "T2M"]

with st.expander("Parameter Cuaca (NASA POWER)"):
    st.markdown(
    """
    - 🌱:rainbow-background[Kelembaban Tanah (0-1)] : Jumlah air dan uap air yang tersedia bagi tanaman di zona perakaran, yang umumnya dianggap sebagai lapisan tanah hingga kedalaman 200 cm, dinyatakan sebagai proporsi air yang terdapat dalam sejumlah tanah tertentu. Nilainya berkisar dari 0 untuk kondisi yang benar-benar kering hingga 1 untuk tanah yang benar-benar jenuh air.
    - 💨:rainbow-background[Kecepatan Angin (m/s)] : Rata-rata kecepatan angin pada ketinggian 10 meter di atas permukaan bumi.
    - 🌍:rainbow-background[Tekanan Permukaan Tanah (kPa)]: Tekanan atmosfer di permukaan bumi.
    - 🌦️:rainbow-background[Presipitasi (mm/hari)] : Rata-rata curah hujan total yang telah dikoreksi bias MERRA-2 di permukaan bumi.
    - 😤:rainbow-background[Kelembaban Udara Relatif (%)] : Rasio tekanan uap terhadap tekanan uap jenuh terhadap permukaan datar air murni, dinyatakan dalam persen.
    - 🌡️:rainbow-background[Temperatur (°C)] : Rasio tekanan uap terhadap tekanan uap jenuh terhadap permukaan datar air murni, dinyatakan dalam persen.
    """
    )

st.write(f'## Total Data: {len(df.columns) * len(df.index)}')
st.write(f'Baris: {len(df)}')

# Cek Data Null
st.write(f'Data Invalid : {(df == -999).sum().sum()}')

df.rename(columns={
    "Kecepatan_Angin": "Kecepatan Angin (m/s)",
    "Tekanan_Permukaan": "Tekanan Permukaan (kPa)",
    "Presipitasi": "Presipitasi (mm/hari)",
    "Kelembaban_Udara": "Kelembaban Udara (%)",
    "Temperatur": "Temperatur (°C)"
}, inplace=True)

st.dataframe(df) 

X = df.values
Y = df.values

split_index = len(st.session_state["X_train"])

# Plotting Pembagian Data Train & Testing
st.markdown("## Visualisasi Data")
plt.figure(figsize=(14, 8))

for i, (column, metric) in enumerate(zip(df.columns, metrics), 1):
    plt.subplot(3, 2, i)
    plt.plot(df.index[:split_index], df[column][:split_index], label='Train')
    plt.plot(df.index[split_index:], df[column][split_index:], label='Test', color='orange')
    plt.xlabel('Tahun')
    plt.ylabel(metric)
    plt.title(column)
    plt.legend()

plt.tight_layout()
st.pyplot(plt.gcf())