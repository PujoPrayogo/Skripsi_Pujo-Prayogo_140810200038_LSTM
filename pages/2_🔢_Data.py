import streamlit as st
import matplotlib.pyplot as plt

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

from streamlit_extras.badges import badge
with st.sidebar:
    st.sidebar.title("⛈️CuacaJakpus")
    badge(type="github", name="PujoPrayogo/Skripsi_Pujo-Prayogo_140810200038_LSTM")
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/id/8/80/Lambang_Universitas_Padjadjaran.svg")
    st.write("©Pujo 2025")

st.title("🔢Data yang digunakan")
st.markdown('Sumber : [POWER NASA](https://power.larc.nasa.gov/data-access-viewer/)')
st.write(f'Rentang Waktu : {st.session_state['start_date'].year} - {st.session_state['end_date'].year}')
st.write(f'Koordinat : {st.session_state['lat']}, {st.session_state['long']}')
st.write('Lokasi : Kota Administrasi Jakarta Pusat')

features_name = [
    "Kelembaban_Tanah",
    "Kecepatan_Angin",
    "Tekanan_Permukaan",
    "Presipitasi",
    "Kelembaban_Udara",
    "Temperatur"
]

features_name_space = [
    "Kelembaban Tanah",
    "Kecepatan Angin",
    "Tekanan Permukaan",
    "Presipitasi",
    "Kelembaban Udara",
    "Temperatur"
]

# List metric yang digunakan pada tiap parameter (berurutan dengan 'features')
metrics = ['0/1', 'm/s', 'kPa', 'mm/d', '%', 'C']

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
    - 🌱:gray-background[Kelembaban Tanah (0-1)] : Jumlah air dan uap air yang tersedia bagi tanaman di zona perakaran, yang umumnya dianggap sebagai lapisan tanah hingga kedalaman 200 cm, dinyatakan sebagai proporsi air yang terdapat dalam sejumlah tanah tertentu. Nilainya berkisar dari 0 untuk kondisi yang benar-benar kering hingga 1 untuk tanah yang benar-benar jenuh air.
    - 💨:gray-background[Kecepatan Angin (m/s)] : Rata-rata kecepatan angin pada ketinggian 10 meter di atas permukaan bumi.
    - 🌦️:gray-background[Presipitasi (mm/d)] : Rata-rata curah hujan total yang telah dikoreksi bias MERRA-2 di permukaan bumi.
    - 🌡️:gray-background[Temperatur (°C)] : rata-rata temperatur udara pada ketinggian 2 meter di atas permukaan bumi.
    - 🌍:gray-background[Tekanan Permukaan Tanah (kPa)]: Tekanan atmosfer di permukaan bumi.
    - 😤:gray-background[Kelembaban Udara Relatif (%)] : Rasio tekanan uap terhadap tekanan uap jenuh terhadap permukaan datar air murni, dinyatakan dalam persen.
    """
    )

st.write(f'## Total Data: {len(df.columns) * len(df.index)}')
st.write(f'Baris: {len(df)}')

# Cek Data Null
st.write(f'Data Invalid : {(df == -999).sum().sum()}')

df.rename(columns={
    "Kelembaban_Tanah": "Kelembaban Tanah (0-1)",
    "Kecepatan_Angin": "Kecepatan Angin (m/s)",
    "Tekanan_Permukaan": "Tekanan Permukaan (kPa)",
    "Presipitasi": "Presipitasi (mm/d)",
    "Kelembaban_Udara": "Kelembaban Udara (%)",
    "Temperatur": "Temperatur (°C)"
}, inplace=True)

# df_table = df
# df_table.index = df_table.index.strftime('%Y-%m-%d')
st.dataframe(df) 

X = df.values
Y = df.values

split_index = len(st.session_state["X_train"])

# Plotting Pembagian Data Train & Testing
def plot_weather_data(df, split_index, column, col):
    """Fungsi untuk menampilkan plot satu variabel cuaca berdasarkan nama kolom df setelah rename."""
    plt.figure(figsize=(12, 5))
    plt.plot(df.index[:split_index], df[column][:split_index], label='Train')
    plt.plot(df.index[split_index:], df[column][split_index:], label='Test', color='orange')
    plt.ylabel(column.split()[-1])  # Mengambil satuan metrik dari nama kolom (misal: "kPa")
    plt.title(column)  # Nama kolom setelah rename sebagai judul
    plt.legend()
    plt.tight_layout()
    col.pyplot(plt.gcf())

st.markdown("## Visualisasi Data")
col1, col2 = st.columns(2)
plot_weather_data(df, split_index, df.columns[0], col1)
plot_weather_data(df, split_index, df.columns[1], col2)
plot_weather_data(df, split_index, df.columns[2], col1)
plot_weather_data(df, split_index, df.columns[3], col2)
plot_weather_data(df, split_index, df.columns[4], col1)
plot_weather_data(df, split_index, df.columns[5], col2)