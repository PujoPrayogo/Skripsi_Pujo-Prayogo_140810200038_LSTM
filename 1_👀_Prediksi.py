import json
import keras
import requests
import numpy as np
import pandas as pd
import streamlit as st
import keras_tuner as kt
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Streamlit extras
from streamlit_extras.badges import badge
from streamlit_extras.let_it_rain import rain 

st.set_page_config(
    page_title="CuacaJakpus",
    page_icon = "⛈️",
)
with st.sidebar:
    st.sidebar.title("⛈️CuacaJakpus")
    badge(type="github", name="PujoPrayogo/Skripsi_Pujo-Prayogo_140810200038_LSTM")
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/id/8/80/Lambang_Universitas_Padjadjaran.svg")
    st.write("©Pujo 2025")

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

st.title('⛈️CuacaJakpus')
st.write('Prediksi Cuaca di Jakarta Pusat menggunakan model LSTM')

target_date = st.date_input('Masukkan Tanggal Prediksi :', datetime(2025, 1, 1))

features = [
    "GWETROOT", 
    "WS10M", 
    "PS", 
    "PRECTOTCORR",
    "RH2M", 
    "T2M"]

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

timesteps = {
    "KT": 30,
    "KA": 20,
    "TP": 40,
    "PR": 30,
    "KU": 20,
    "T": 20
}

metrics_dict = {
    "GWETROOT": "", #Hanya 0-1
    "WS10M": "m/s",
    "PS": "kPa",
    "PRECTOTCORR": "mm/day",
    "RH2M": "C",
    "T2M": "g/kg"
}

# ----- Tanggal dan Koordinat lokasi data pada POWER NASA
start_date = "20200101"
end_date = "20241231"
lat = "-6.18"
long = "106.83"

if 'start_date' not in st.session_state:
    st.session_state['start_date'] = datetime.strptime(start_date, "%Y%m%d")

if 'end_date' not in st.session_state:
    st.session_state['end_date'] = datetime.strptime(end_date, "%Y%m%d")

if 'lat' not in st.session_state:
    st.session_state["lat"] = -6.18

if 'long' not in st.session_state:
    st.session_state["long"] = 106.83

# -------------------------------------------------------------------------------------- FUNGSI (5)
# ----- Fungsi Membuat Sequence 
def create_sequences(X, timestep):
    X_seq, y_seq = [], []
    for i in range(len(X) - timestep):
        X_seq.append(X[i:i + timestep])
        y_seq.append(X[i + timestep])
    return np.array(X_seq), np.array(y_seq)

# ----- Fungsi Prediksi
def predict_until(target_date, model, data, scaler, timesteps, initial_date):
    # ----- Hitung jumlah hari dari tanggal terakhir data hingga target_date
    future_dates = pd.date_range(start=initial_date + timedelta(days=1), end=target_date)
    future_steps = len(future_dates)  # Hitung semua langkah prediksi

    # ----- Inisialisasi input data untuk prediksi awal
    current_input = data.reshape(1, timesteps, 1)  # Reshape input untuk LSTM
    
    # ----- Prediksi berulang untuk setiap hari di masa depan
    future_predictions = []
    for _ in range(future_steps):
        pred = model.predict(current_input)
        future_predictions.append(pred[0])
        
        # ----- Update input sequence dengan prediksi baru
        current_input = np.append(current_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    
    # ----- Kembalikan prediksi ke skala asli
    future_predictions_inv = scaler.inverse_transform(future_predictions)
    
    return future_dates, future_predictions_inv

# ----- Fungsi Plot hasil Prediksi
def prediction_plot(df, X_train, timesteps, y_test_inv, future_dates, future_preds, feature, feature_name, target_date, col):
    # --- FILTER DATA AKTUAL HANYA BULAN DESEMBER 2024 ---
    last_actual_date = pd.to_datetime('2024-12-01')  # Tanggal awal untuk data aktual yang ditampilkan
    actual_index = df.index[len(X_train) + timesteps:]  # Index asli data aktual
    mask = actual_index >= last_actual_date  # Filter hanya untuk bulan Desember 2024 ke depan

    # --- VISUALISASI HASIL PREDIKSI ---
    plt.figure(figsize=(10, 4))
    
    # Plot hanya data aktual bulan Desember 2024 ke depan
    plt.plot(actual_index[mask], y_test_inv[mask], label='Data Aktual', color='blue')

    # Plot prediksi masa depan
    plt.plot(future_dates, future_preds, label='Prediksi Masa Depan', color='red', linestyle='-')

    # Anotasi prediksi pada target_date
    plt.annotate(f'{future_preds[-1][0]:.2f} {metrics_dict[feature]}',
                 xy=(future_dates[-1], future_preds[-1][0]),
                 xytext=(5, 10),
                 textcoords='offset points')

    plt.xlabel('Tahun')
    plt.ylabel(metrics_dict[feature])
    plt.title(f'Prediksi {feature_name} Hingga {target_date}')
    plt.grid(True)
    plt.legend()
    col.pyplot(plt.gcf())

# ----- Fungsi Evaluasi Model
def evaluasi(y_test_inv, y_pred_inv):
    r2 = r2_score(y_test_inv, y_pred_inv)
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    return r2, mape

# ----- Fungsi Evaluasi Model (untuk presipitasi)
def evaluasi_presipitasi(y_test_inv, y_pred_inv):  # Khusus untuk Presipitasi karena nilainya mendekati 0
    r2 = r2_score(y_test_inv, y_pred_inv)
    smape = np.mean(2 * np.abs(y_test_inv - y_pred_inv) / (np.abs(y_test_inv) + np.abs(y_pred_inv))) * 100
    return r2, smape

# ----- Download Data melalui API POWER NASA
with st.spinner("Sedang Mengambil data..."):
    # URL Request
    request_url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?"
                f"start={start_date}&end={end_date}&latitude={lat}&longitude={long}&community=ag&"
                f"parameters={','.join(features)}&format=json&user=pujo&header=true&time-standard=lst")

    response = requests.get(url=request_url, verify=True, timeout=30.00)
    content = json.loads(response.content.decode('utf-8'))
    df_all = pd.DataFrame.from_dict(content['properties']['parameter'])
    df_all.index = pd.to_datetime(df_all.index, format='%Y%m%d')

    df_all.rename(columns={'GWETROOT': 'Kelembaban_Tanah',
                    'WS10M': 'Kecepatan_Angin',
                    'PS': 'Tekanan_Permukaan',
                    'PRECTOTCORR': 'Presipitasi',
                    'RH2M': 'Kelembaban_Udara',
                    'T2M': 'Temperatur'}, inplace=True)
    if 'df_all' not in st.session_state:
        st.session_state["df_all"] = df_all

    # Buat dictionary untuk menyimpan setiap DataFrame berdasarkan nama kolom
    df_dict = {col: df_all[[col]].copy() for col in df_all.columns}

    # Akses DataFrame dengan nama yang sesuai
    df_KT = df_dict['Kelembaban_Tanah']
    df_KA = df_dict['Kecepatan_Angin']
    df_TP = df_dict['Tekanan_Permukaan']
    df_PR = df_dict['Presipitasi']
    df_KU = df_dict['Kelembaban_Udara']
    df_T  = df_dict['Temperatur']

with st.spinner("Sedang Sequencing..."):
    # ----- Scaling data dengan MinMax
    KT_scaler = MinMaxScaler()
    KA_scaler = MinMaxScaler()
    TP_scaler = MinMaxScaler()
    PR_scaler = MinMaxScaler()
    KU_scaler = MinMaxScaler()
    T_scaler = MinMaxScaler()

    scaled_KT_data = KT_scaler.fit_transform(df_KT[[features_name[0]]])
    scaled_KA_data = KA_scaler.fit_transform(df_KA[[features_name[1]]])
    scaled_TP_data = TP_scaler.fit_transform(df_TP[[features_name[2]]])
    scaled_PR_data = PR_scaler.fit_transform(df_PR[[features_name[3]]])
    scaled_KU_data = KU_scaler.fit_transform(df_KU[[features_name[4]]])
    scaled_T_data = T_scaler.fit_transform(df_T[[features_name[5]]])

    # Train-Test Split (8:2)
    X_KT_train, X_KT_test, y_KT_train, y_KT_test = train_test_split(
        scaled_KT_data, scaled_KT_data, test_size=0.2, shuffle=False
    )
    X_KA_train, X_KA_test, y_KA_train, y_KA_test = train_test_split(
        scaled_KA_data, scaled_KA_data, test_size=0.2, shuffle=False
    )
    X_TP_train, X_TP_test, y_TP_train, y_TP_test = train_test_split(
        scaled_TP_data, scaled_TP_data, test_size=0.2, shuffle=False
    )
    X_PR_train, X_PR_test, y_PR_train, y_PR_test = train_test_split(
        scaled_PR_data, scaled_PR_data, test_size=0.2, shuffle=False
    )
    X_KU_train, X_KU_test, y_KU_train, y_KU_test = train_test_split(
        scaled_KU_data, scaled_KU_data, test_size=0.2, shuffle=False
    )
    X_T_train, X_T_test, y_T_train, y_T_test = train_test_split(
        scaled_T_data, scaled_T_data, test_size=0.2, shuffle=False
    )

    if 'X_train' not in st.session_state:
        st.session_state["X_train"] = X_KT_train

    # -------------------------------------------------------------------------------------------- KT
    X_KT_train_seq, y_KT_train_seq = create_sequences(X_KT_train, timesteps["KT"])
    X_KT_test_seq, y_KT_test_seq = create_sequences(X_KT_test, timesteps["KT"])

    # ----- Reshape 
    X_KT_train_seq = X_KT_train_seq.reshape((X_KT_train_seq.shape[0], X_KT_train_seq.shape[1], 1))
    X_KT_test_seq = X_KT_test_seq.reshape((X_KT_test_seq.shape[0], X_KT_test_seq.shape[1], 1))

    # -------------------------------------------------------------------------------------------- KA
    X_KA_train_seq, y_KA_train_seq = create_sequences(X_KA_train, timesteps["KA"])
    X_KA_test_seq, y_KA_test_seq = create_sequences(X_KA_test, timesteps["KA"])

    # ----- Reshape 
    X_KA_train_seq = X_KA_train_seq.reshape((X_KA_train_seq.shape[0], X_KA_train_seq.shape[1], 1))
    X_KA_test_seq = X_KA_test_seq.reshape((X_KA_test_seq.shape[0], X_KA_test_seq.shape[1], 1))


    # -------------------------------------------------------------------------------------------- TP
    X_TP_train_seq, y_TP_train_seq = create_sequences(X_TP_train, timesteps["TP"])
    X_TP_test_seq, y_TP_test_seq = create_sequences(X_TP_test, timesteps["TP"])

    # ----- Reshape 
    X_TP_train_seq = X_TP_train_seq.reshape((X_TP_train_seq.shape[0], X_TP_train_seq.shape[1], 1))
    X_TP_test_seq = X_TP_test_seq.reshape((X_TP_test_seq.shape[0], X_TP_test_seq.shape[1], 1))


    # -------------------------------------------------------------------------------------------- PR
    X_PR_train_seq, y_PR_train_seq = create_sequences(X_PR_train, timesteps["PR"])
    X_PR_test_seq, y_PR_test_seq = create_sequences(X_PR_test, timesteps["PR"])

    # ----- Reshape 
    X_PR_train_seq = X_PR_train_seq.reshape((X_PR_train_seq.shape[0], X_PR_train_seq.shape[1], 1))
    X_PR_test_seq = X_PR_test_seq.reshape((X_PR_test_seq.shape[0], X_PR_test_seq.shape[1], 1))


    # -------------------------------------------------------------------------------------------- KU 
    X_KU_train_seq, y_KU_train_seq = create_sequences(X_KU_train, timesteps["KU"])
    X_KU_test_seq, y_KU_test_seq = create_sequences(X_KU_test, timesteps["KU"])

    # ----- Reshape 
    X_KU_train_seq = X_KU_train_seq.reshape((X_KU_train_seq.shape[0], X_KU_train_seq.shape[1], 1))
    X_KU_test_seq = X_KU_test_seq.reshape((X_KU_test_seq.shape[0], X_KU_test_seq.shape[1], 1))


    # -------------------------------------------------------------------------------------------- T
    X_T_train_seq, y_T_train_seq = create_sequences(X_T_train, timesteps["T"])
    X_T_test_seq, y_T_test_seq = create_sequences(X_T_test, timesteps["T"])

    # ----- Reshape 
    X_T_train_seq = X_T_train_seq.reshape((X_T_train_seq.shape[0], X_T_train_seq.shape[1], 1))
    X_T_test_seq = X_T_test_seq.reshape((X_T_test_seq.shape[0], X_T_test_seq.shape[1], 1))

with st.spinner("Sedang load model..."):
    # ----- Load model dari file .keras
    model_KT = keras.saving.load_model(f'Kelembaban_Tanah_model.keras')
    model_KA = keras.saving.load_model(f'Kecepatan_Angin_model.keras')
    model_TP = keras.saving.load_model(f'Tekanan_Permukaan_model.keras')
    model_PR = keras.saving.load_model(f'Presipitasi_model.keras')
    model_KU = keras.saving.load_model(f'Kelembaban_Udara_model.keras')
    model_T = keras.saving.load_model(f'Temperatur_model.keras')

    # ----- Prediksi dengan model -------------------------- KT
    y_KT_pred = model_KT.predict(X_KT_test_seq)
    y_KT_pred_inv = KT_scaler.inverse_transform(y_KT_pred)   # Invers Skala Prediksi
    y_KT_test_inv = KT_scaler.inverse_transform(y_KT_test_seq)

    # ----- Prediksi dengan model -------------------------- KA
    y_KA_pred = model_KA.predict(X_KA_test_seq)
    y_KA_pred_inv = KA_scaler.inverse_transform(y_KA_pred)
    y_KA_test_inv = KA_scaler.inverse_transform(y_KA_test_seq)

    # ----- Prediksi dengan model -------------------------- TP
    y_TP_pred = model_TP.predict(X_TP_test_seq)
    y_TP_pred_inv = TP_scaler.inverse_transform(y_TP_pred)
    y_TP_test_inv = TP_scaler.inverse_transform(y_TP_test_seq)

    # ----- Prediksi dengan model -------------------------- PR
    y_PR_pred = model_PR.predict(X_PR_test_seq)
    y_PR_pred_inv = PR_scaler.inverse_transform(y_PR_pred)
    y_PR_test_inv = PR_scaler.inverse_transform(y_PR_test_seq)

    # ----- Prediksi dengan model -------------------------- KU
    y_KU_pred = model_KU.predict(X_KU_test_seq)
    y_KU_pred_inv = KU_scaler.inverse_transform(y_KU_pred)
    y_KU_test_inv = KU_scaler.inverse_transform(y_KU_test_seq)

    # ----- Prediksi dengan model -------------------------- T
    y_T_pred = model_T.predict(X_T_test_seq)
    y_T_pred_inv = T_scaler.inverse_transform(y_T_pred)
    y_T_test_inv = T_scaler.inverse_transform(y_T_test_seq)

# ------------------------------------------------------------------------------------- MAIN APPLICATION
with st.spinner("Sedang Memprediksi...",  show_time=True):
    KT_future_dates, KT_future_preds = predict_until(target_date, model_KT, X_KT_test_seq[-1], KT_scaler, timesteps["KT"], df_KT.index[-1])
    KA_future_dates, KA_future_preds = predict_until(target_date, model_KA, X_KA_test_seq[-1], KA_scaler, timesteps["KA"], df_KA.index[-1])
    TP_future_dates, TP_future_preds = predict_until(target_date, model_TP, X_TP_test_seq[-1], TP_scaler, timesteps["TP"], df_TP.index[-1])
    PR_future_dates, PR_future_preds = predict_until(target_date, model_PR, X_PR_test_seq[-1], PR_scaler, timesteps["PR"], df_PR.index[-1])
    KU_future_dates, KU_future_preds = predict_until(target_date, model_KU, X_KU_test_seq[-1], KU_scaler, timesteps["KU"], df_KU.index[-1])
    T_future_dates, T_future_preds = predict_until(target_date, model_T, X_T_test_seq[-1], T_scaler, timesteps["T"], df_T.index[-1])

st.success("Selesai!")
st.balloons()

# ----- Menggabungkan hasil prediksi masa depan dengan data prediksi sebelumnya ------------ KT
KT_combined_dates = np.concatenate([df_KT.index[len(X_KT_train) + timesteps["KT"]:], KT_future_dates])
KT_combined_predictions = np.concatenate([y_KT_pred_inv, KT_future_preds])   

# ----- Menggabungkan hasil prediksi masa depan dengan data prediksi sebelumnya ------------ KA
KA_combined_dates = np.concatenate([df_KA.index[len(X_KA_train) + timesteps["KA"]:], KA_future_dates])
KA_combined_predictions = np.concatenate([y_KA_pred_inv, KA_future_preds])   


# ----- Menggabungkan hasil prediksi masa depan dengan data prediksi sebelumnya ------------ TP
TP_combined_dates = np.concatenate([df_TP.index[len(X_TP_train) + timesteps["TP"]:], TP_future_dates])
TP_combined_predictions = np.concatenate([y_TP_pred_inv, TP_future_preds])   

# ----- Menggabungkan hasil prediksi masa depan dengan data prediksi sebelumnya ------------ PR
PR_combined_dates = np.concatenate([df_PR.index[len(X_PR_train) + timesteps["PR"]:], PR_future_dates])
PR_combined_predictions = np.concatenate([y_PR_pred_inv, PR_future_preds])   


# ----- Menggabungkan hasil prediksi masa depan dengan data prediksi sebelumnya ------------ KU
KU_combined_dates = np.concatenate([df_KU.index[len(X_KU_train) + timesteps["KU"]:], KU_future_dates])
KU_combined_predictions = np.concatenate([y_KU_pred_inv, KU_future_preds])   


# ----- Menggabungkan hasil prediksi masa depan dengan data prediksi sebelumnya ------------ T
T_combined_dates = np.concatenate([df_T.index[len(X_T_train) + timesteps["T"]:], T_future_dates])
T_combined_predictions = np.concatenate([y_T_pred_inv, T_future_preds])   

col1, col2, col3 = st.columns(3)
col1.metric(":rainbow-background[Kelembaban Tanah]", f"🌱{np.round(KT_future_preds[-1], 2)[0]}")
col2.metric(":rainbow-background[Kecepatan Angin]", f"💨{np.round(KA_future_preds[-1], 2)[0]} m/s")
col2.metric(":rainbow-background[Tekanan Permukaan]",f"🌍{np.round(TP_future_preds[-1], 2)[0]} kPa")
col3.metric(":rainbow-background[Presipitasi]", f"🌦️{np.round(PR_future_preds[-1], 2)[0]} mm/d")
col3.metric(":rainbow-background[Kelembaban Udara]", f"😤{np.round(KU_future_preds[-1], 2)[0]} %")
col1.metric(":rainbow-background[Temperatur]", f'🌡️{np.round(T_future_preds[-1], 2)[0]} °C')

st.divider()
col1, col2 = st.columns(2)
# ----- Print Hasil Prediksi
prediksi = [ KT_future_preds[-1], KA_future_preds[-1], TP_future_preds[-1], PR_future_preds[-1] , KU_future_preds[-1], T_future_preds[-1] ]
prediksi_dict = {'Parameter' : features_name_space, f'prediksi {target_date}' : prediksi}
prediksi_df = pd.DataFrame(prediksi_dict)
prediksi_df.index += 1
col1.write("## Data")
col1.write(prediksi_df)

# ----- Print Akurasi Model
KT_r2, KT_mape = evaluasi(y_KT_test_inv, y_KT_pred_inv)
KA_r2, KA_mape = evaluasi(y_KA_test_inv, y_KA_pred_inv)
TP_r2, TP_mape = evaluasi(y_TP_test_inv, y_TP_pred_inv)
PR_r2, PR_mape = evaluasi_presipitasi(y_PR_test_inv, y_PR_pred_inv)
KU_r2, KU_mape = evaluasi(y_KU_test_inv, y_KU_pred_inv)
T_r2, T_mape = evaluasi(y_T_test_inv, y_T_pred_inv)

r2 = [KT_r2, KA_r2, TP_r2, PR_r2, KU_r2, T_r2]
mape = [KT_mape, KA_mape, TP_mape, PR_mape, KU_mape, T_mape]

akurasi_dict = {'Parameter' : features_name_space, 'R2' : r2, ' MAPE%' : mape}
akurasi_df = pd.DataFrame(akurasi_dict)
akurasi_df.index += 1
col2.write("## Akurasi")
col2.write(akurasi_df)
#col2.write(akurasi_df.to_html(index=False), unsafe_allow_html=True)

col = st.columns(1)
st.write("## Grafik")

col1, col2 = st.columns(2)

with st.expander("Grafik Prediksi"):
    prediction_plot(df_KT, X_KT_train, timesteps["KT"], y_KT_test_inv, KT_future_dates, KT_future_preds, features[0], features_name_space[0], target_date, col1)
    prediction_plot(df_KA, X_KA_train, timesteps["KA"], y_KA_test_inv, KA_future_dates, KA_future_preds, features[1], features_name_space[1], target_date, col2)
    prediction_plot(df_TP, X_TP_train, timesteps["TP"], y_TP_test_inv, TP_future_dates, TP_future_preds, features[2], features_name_space[2], target_date, col1)
    prediction_plot(df_PR, X_PR_train, timesteps["PR"], y_PR_test_inv, PR_future_dates, PR_future_preds, features[3], features_name_space[3], target_date, col2)
    prediction_plot(df_KU, X_KU_train, timesteps["KU"], y_KU_test_inv, KU_future_dates, KU_future_preds, features[4], features_name_space[4], target_date, col1)
    prediction_plot(df_T, X_T_train, timesteps["T"], y_T_test_inv, T_future_dates, T_future_preds, features[5], features_name_space[5], target_date, col2)