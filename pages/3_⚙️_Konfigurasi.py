import streamlit as st
import keras_tuner as kt
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.stattools import acf
from streamlit_extras.let_it_rain import rain 

st.set_page_config(
    page_title="Konfigurasi Model CuacaJakpus",
    page_icon = "⛈️",
)
st.title("⚙️Konfigurasi Model")

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

features_name = [
    "Kelembaban_Tanah",
    "Kecepatan_Angin",
    "Tekanan_Permukaan",
    "Presipitasi",
    "Kelembaban_Udara",
    "Temperatur"
]

features = [
    "GWETROOT", 
    "WS10M", 
    "PS", 
    "PRECTOTCORR",
    "RH2M", 
    "T2M"]

def build_model(hp):
    model = Sequential()

    # ------ Jumlah Layers, jumlah unit, fungsi aktivasi
    for i in range(hp.Choice('layers', [1, 2])):  # Tuning layer LSTM (2)
        model.add(LSTM(
            units=hp.Choice('filters', [32, 64, 128]),  # Tuning jumlah unit pada layer LSTM (3) 
            activation=hp.Choice('activation', ['relu', 'tanh', 'sigmoid']),  # Tuning fungsi aktivasi (3)
            return_sequences=True if i < hp.Choice('layers', [1, 2]) - 1 else False  
        ))
        model.add(Dropout(0.2)) # Dropout
    model.add(Dense(1))
    
    # ----- Optimizer dan Learning Rate
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop']) # Tuning Optimizer (2)
    learning_rate = hp.Choice('learning_rate', [0.01, 0.001]) # Tuning learning Rate (2)
     
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)
    
    # ----- Compile Model
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    return model

def bestTuningResults(feature_name):
    # ----- Inisiasi Tuner 
    tuner = kt.GridSearch(
        build_model,
        objective='val_loss',
        max_trials=72,
        executions_per_trial=1,
        directory=f'lstm_tuning_{feature_name}_dir',
        project_name=f'lstm_{feature_name}_tuning'
    )

    # ----- Output Best Hyperparameter
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # ----- Ambil Best Trial untuk mendapatkan nilai skor (MAE)
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    best_score = best_trial.score  # Nilai MAE dari trial terbaik

    # ----- Menyimpan hasil tuning terbaik sebagai dictionary
    best_params = {
        "Feature": feature_name,
        "Layers": best_hps.get('layers'),
        "Filters": best_hps.get('filters'),
        "Activation": best_hps.get('activation'),
        "Optimizer": best_hps.get('optimizer'),
        "Learning Rate": best_hps.get('learning_rate'),
        "Score (MAE)": best_score  # Menyimpan skor terbaik
    }
    
    return best_params

# ----- Timesteps
timesteps = {
    "Kelembaban_Tanah": 30,
    "Kecepatan_Angin": 20,
    "Tekanan_Permukaan": 40,
    "Presipitasi": 30,
    "Kelembaban_Udara": 30,
    "Temperatur": 20
}

features_name_space = [
    "Kelembaban Tanah",
    "Kecepatan Angin",
    "Tekanan Permukaan",
    "Presipitasi",
    "Kelembaban Udara",
    "Temperatur"
]

with st.spinner("Sedang Load Tuning..."):
# ----- Menggabungkan hasil tuning terbaik ke dalam satu DataFrame
    df_best_results = pd.DataFrame([bestTuningResults(feature) for feature in features_name])
    df_best_results["Timesteps"] = df_best_results["Feature"].map(timesteps)
    df_best_results.index += 1

    # ----- Menampilkan seluruh hasil terbaik dalam satu tabel
    st.write("## Konfigurasi Hyperparameter Terbaik")
    st.write(df_best_results)

def tuningResult(feature_name):
    # ----- Inisiasi Tuner 
    tuner = kt.GridSearch(
        build_model,
        objective='val_loss',
        max_trials=72,  # Jumlah Kombinasi Hyperparameter
        executions_per_trial=1,  # 2x training per kombinasi hyperparameter
        directory=f'lstm_tuning_{feature_name}_dir',
        project_name=f'lstm_{feature_name}_tuning'
    )

    # ----- Output Best Hyperparameter
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    tuner_results = tuner.oracle.get_best_trials(num_trials=72)  
    results = []

    # ----- Memasukkan Hasil Tuning ke DataFrame
    for trial in tuner_results:
        values = trial.hyperparameters.values
        values['score'] = trial.score  # Nilai skor (mae)
        values['layers'] = trial.hyperparameters.get('layers')  # Menambahkan jumlah layer
        results.append(values)

    df_results = pd.DataFrame(results)
    df_results.index += 1

    # ----- Print seluruh hasil hyperparameter tuning
    st.write(df_results)

# def prediction_plot(df, X_train, timesteps, y_test_inv, y_pred_inv, feature, feature_name, target_date, col):
#     # --- VISUALISASI HASIL PREDIKSI ---
#     plt.figure(figsize=(10, 4))
#     plt.plot(df.index[len(X_train) + timesteps:], y_test_inv, label='Data Aktual', color='blue')
#     plt.plot(df.index[len(X_train) + timesteps:], y_pred_inv, label='Prediksi Model', color='orange')

#     plt.xlabel('Tahun')
#     plt.ylabel(metrics_dict[feature])
#     plt.title(f'Prediksi {feature_name} Hingga {target_date}')
#     plt.grid(True)
#     plt.legend()
#     col.pyplot(plt.gcf())

# ----- Fungsi Autokorelasi
def determine_timesteps(df, max_lag, features_name_space):
    timesteps_dict = {}
    
    for i, column in enumerate(df.columns):
        # Menghitung autokorelasi hingga max_lag
        lag_acf = acf(df[column], nlags=max_lag)
        st.markdown(f"## {features_name_space[i]}")
        
        # Plotting autokorelasi
        plt.figure(figsize=(10, 6))
        plt.stem(range(len(lag_acf)), lag_acf)
        #plt.title(f'Autokorelasi Data {features_name_space[i]}')
        plt.xlabel('Lag')
        plt.ylabel('Autokorelasi')
        st.pyplot(plt.gcf())
    return timesteps_dict

with st.expander("Autokorelasi"):
    max_lags = st.slider("Jumlah lags", 0, 100, 50)

    # Menjalankan fungsi untuk semua parameter dalam DataFrame
    timesteps_dict = determine_timesteps(st.session_state["df_all"], max_lags, features_name_space)

with st.expander("Hasil Tuning"):
    st.write("Opsi Hyperparameter")
    hyperparameters = pd.DataFrame(
        {
            "Hyperparameter": ["Layers", "Filters", "Activation", "Optimizer", "Learning Rate"],
            "Nilai": ["1, 2", "relu, tanh, sigmoid", "32, 64, 128", "0,001 ; 0,01", "adam, rmsprop"]
        }
    )
    hyperparameters.index +=1
    st.table(hyperparameters)
    
    for i in range(6):
        st.write(f"## {features_name_space[i]}")
        tuningResult(features_name[i])
        st.divider()