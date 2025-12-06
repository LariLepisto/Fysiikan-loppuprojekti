import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import folium
from streamlit_folium import st_folium


# --------------------------------------------------
# APUTOIMINNOT
# --------------------------------------------------

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    
    if high >= 1:
        high = 0.99
    if low <= 0:
        low = 0.001

    b, a = signal.butter(order, [low, high], btype="band")
    y = signal.filtfilt(b, a, data)
    return y

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Laskee kahden pisteen välisen etäisyyden (metreinä).
    """
    R = 6371000.0  # Maapallon säde metreinä
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d  # metriä

def compute_gps_distance_and_speed(df_loc):
    """
    Palauttaa (distance_m, avg_speed_m_s, total_time_s).
    """
    time = df_loc.iloc[:, 0].values  # oletus: ensimmäinen sarake on aika sekunteina
    total_time = time[-1] - time[0]

    # sarakenimet etsimistä varten
    cols_lower = [c.lower() for c in df_loc.columns]

    lat_col = None
    lon_col = None
    speed_col = None

    for c in df_loc.columns:
        cl = c.lower()
        if "lat" in cl and lat_col is None:
            lat_col = c
        if "lon" in cl and lon_col is None:
            lon_col = c
        if "speed" in cl and speed_col is None:
            speed_col = c

    distance = 0.0

    if speed_col is not None:
        # käytä nopeutta ja aikavälejä
        speed_values = df_loc[speed_col].dropna()
        if len(speed_values) > 0:
            speed = df_loc[speed_col].values  # m/s oletus
            dt = np.diff(time)
            # käytetään nopeutta edellisessä intervallissa
            distance = np.sum(speed[:-1] * dt)
        else:
            lat = df_loc[lat_col].values
            lon = df_loc[lon_col].values
            # laske etäisyydet pisteiden välillä
            dists = haversine_distance(lat[:-1], lon[:-1], lat[1:], lon[1:])
            distance = np.sum(dists)
    elif lat_col is not None and lon_col is not None:
        lat = df_loc[lat_col].values
        lon = df_loc[lon_col].values
        # laske etäisyydet pisteiden välillä
        dists = haversine_distance(lat[:-1], lon[:-1], lat[1:], lon[1:])
        distance = np.sum(dists)
    else:
        distance = np.nan

    if total_time > 0:
        avg_speed = distance / total_time
    else:
        avg_speed = np.nan

    return distance, avg_speed, total_time

def compute_steps_time_domain(acc, fs, height=None, distance_samples=None):
    """
    Laskee askeleet huippujen perusteella.
    height: minimihuippukorkeus
    distance_samples: minimietäisyys huippujen välillä (näytteissä)
    """
    if height is None:
        # karkea oletus: otsikoidaan huippu, jos yli (keskiarvo + 0.5 * std)
        height = np.mean(acc) + 0.5 * np.std(acc)

    if distance_samples is None:
        # oletetaan max 4 askelta sekunnissa -> vähintään fs/4 näytettä huippujen välissä
        distance_samples = int(fs / 4)

    peaks, _ = signal.find_peaks(acc, height=height, distance=distance_samples)
    return peaks

def compute_steps_fft(acc, fs, fmin=0.5, fmax=4.0):
    """
    Laskee askelmäärän Fourier-analyysillä (PSD / Welch).
    Palauttaa (step_count_fft, freqs, psd, dominant_freq).
    """
    # poista DC-komponentti
    acc_detrended = acc - np.mean(acc)

    n = len(acc_detrended)
    if n < 10:
        return np.nan, None, None, None

    freqs, psd = signal.welch(acc_detrended, fs=fs, nperseg=min(256, n))

    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan, freqs, psd, None

    dominant_idx = np.argmax(psd[mask])
    dominant_freq = freqs[mask][dominant_idx]

    duration = n / fs
    step_count_fft = dominant_freq * duration

    return step_count_fft, freqs, psd, dominant_freq


# --------------------------------------------------
# STREAMLIT-SOVELLUS
# --------------------------------------------------

st.set_page_config(page_title="GPS + kiihtyvyys -analyysi", layout="wide")

st.title("📱 GPS + kiihtyvyys -liikeanalyysi")
st.write(
    """
Tämä sovellus analysoi samanaikaisesti mitattua kiihtyvyys- ja GPS-dataa.
Saat talteen:

- Askelmäärän **suodatetusta kiihtyvyysdatasta** (huippujen laskeminen)
- Askelmäärän **Fourier-analyysin perusteella**
- GPS:stä lasketun **matkan** ja **keskinopeuden**
- Näistä johdetun **askelpituuden**
- Kuvakkeen: suodatettu kiihtyvyys, tehospektritiheys ja reitti kartalla
"""
)

# --------------------------------------------------
# DATAN LUKEMINEN KANSIOSTA
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ACC_PATH = os.path.join(DATA_DIR, "Accelerometer.csv")  # Iso A ja L
LOC_PATH = os.path.join(DATA_DIR, "Location.csv")

st.sidebar.header("1. Data")
st.sidebar.info(f"Luetaan dataa kansiosta:\n`{DATA_DIR}`")

# Diagnostiikka
if not os.path.exists(ACC_PATH):
    st.sidebar.error(f"❌ accelerometer.csv puuttuu: {ACC_PATH}")
    st.stop()
if not os.path.exists(LOC_PATH):
    st.sidebar.error(f"❌ location.csv puuttuu: {LOC_PATH}")
    st.stop()

st.sidebar.success("✅ Molemmat tiedostot löytyi!")

# Lue tiedostot
df_acc = pd.read_csv(ACC_PATH)
df_loc = pd.read_csv(LOC_PATH)

# ✅ LISÄÄ TÄMÄ TÄSSÄ - heti datan lukemisen jälkeen
acc_components = {
    "X (m/s^2)": df_acc["X (m/s^2)"].values if "X (m/s^2)" in df_acc.columns else None,
    "Y (m/s^2)": df_acc["Y (m/s^2)"].values if "Y (m/s^2)" in df_acc.columns else None,
    "Z (m/s^2)": df_acc["Z (m/s^2)"].values if "Z (m/s^2)" in df_acc.columns else None,
}

st.sidebar.header("2. Asetukset")

fs = st.sidebar.number_input(
    "Kiihtyvyysdatan näytteenottotaajuus [Hz]",
    min_value=1.0,
    max_value=200.0,
    value=50.0,
    step=1.0
)

lowcut = st.sidebar.slider("Suotimen alarajataajuus [Hz]", 0.1, 5.0, 0.5, 0.1)
highcut = st.sidebar.slider("Suotimen ylärajataajuus [Hz]", 1.0, 15.0, 5.0, 0.5)

st.sidebar.write("Huippujen etsinnän asetukset (askelmäärä 1):")
user_peak_height = st.sidebar.slider("Minimihuipun korkeus (relatiivinen)", 0.0, 3.0, 0.5, 0.1)
min_step_freq = st.sidebar.slider("Minimitaajuus [Hz] askelille (FFT)", 0.5, 4.0, 0.5, 0.1)
max_step_freq = st.sidebar.slider("Maksimitaajuus [Hz] askelille (FFT)", 1.0, 6.0, 4.0, 0.1)

# --------------------------------------------------
# RAAKADATAN ESIKATSELU
# --------------------------------------------------

st.subheader("Raakadatan esikatselu")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Kiihtyvyysdata (ensimmäiset rivit)**")
    st.dataframe(df_acc.head())
with c2:
    st.markdown("**GPS-data (ensimmäiset rivit)**")
    st.dataframe(df_loc.head())

# --------------------------------------------------
# VALITAAN KIIHTYVYYKSIEN KOMPONENTTI
# --------------------------------------------------

acc_columns = [
    col for col in df_acc.columns[1:]  # oletus: eka sarake on aika
    if "acc" in col.lower() or "linear" in col.lower() or "m/s" in col.lower()
]

if not acc_columns:
    acc_columns = list(df_acc.columns[1:])  # fallback

st.subheader("Kiihtyvyyden komponentin valinta")
selected_acc_col = st.selectbox(
    "Valitse kiihtyvyyden komponentti analyysiin",
    acc_columns
)

time_acc = df_acc.iloc[:, 0].values
acc_raw = df_acc[selected_acc_col].values

# Lasketaan efektiivinen fs, jos käyttäjä haluaa
if len(time_acc) > 1:
    dt_est = np.median(np.diff(time_acc))
    if dt_est > 0:
        fs_est = 1.0 / dt_est
    else:
        fs_est = fs
else:
    fs_est = fs

st.caption(f"Arvioitu näytteenottotaajuus datasta: ~{fs_est:.2f} Hz (käytetään asetuksissa annettua {fs:.2f} Hz laskuissa).")

# --------------------------------------------------
# SUODATUS
# --------------------------------------------------

acc_filtered = butter_bandpass_filter(acc_raw, lowcut, highcut, fs)

# --------------------------------------------------
# ASKELMÄÄRÄ 1: HUIPPUJEN LASKEMINEN
# --------------------------------------------------

# peak_height asetetaan suhteessa suodatetun signaalin keskihajontaan
peak_height_absolute = np.mean(acc_filtered) + user_peak_height * np.std(acc_filtered)
distance_samples = int(fs / 4)

peaks = compute_steps_time_domain(
    acc_filtered,
    fs,
    height=peak_height_absolute,
    distance_samples=distance_samples
)
step_count_time = len(peaks)

# --------------------------------------------------
# ASKELMÄÄRÄ 2: FOURIER/PSD
# --------------------------------------------------

step_count_fft, freqs, psd, dominant_freq = compute_steps_fft(
    acc_filtered,
    fs,
    fmin=min_step_freq,
    fmax=max_step_freq
)

# --------------------------------------------------
# GPS: MATKA, KESKINOPEUS
# --------------------------------------------------

distance_m, avg_speed_m_s, total_time_s = compute_gps_distance_and_speed(df_loc)

# --------------------------------------------------
# ASKELPITUUS
# --------------------------------------------------

if step_count_time > 0 and not np.isnan(distance_m):
    step_length_m = distance_m / step_count_time
else:
    step_length_m = np.nan

# --------------------------------------------------
# NÄYTÄ TULOKSET
# --------------------------------------------------

st.subheader("Yhteenveto")

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Askelmäärä (huiput)", f"{step_count_time:d}")
with m2:
    if not np.isnan(step_count_fft):
        st.metric("Askelmäärä (FFT, arvio)", f"{step_count_fft:.1f}")
    else:
        st.metric("Askelmäärä (FFT, arvio)", "–")
with m3:
    if not np.isnan(distance_m):
        st.metric("Matka", f"{distance_m/1000:.3f} km")
    else:
        st.metric("Matka", "–")
with m4:
    if not np.isnan(avg_speed_m_s):
        st.metric("Keskinopeus", f"{avg_speed_m_s:.2f} m/s")
    else:
        st.metric("Keskinopeus", "–")
with m5:
    if not np.isnan(step_length_m):
        st.metric("Askelpituus", f"{step_length_m:.2f} m")
    else:
        st.metric("Askelpituus", "–")

st.caption(f"Kokonaisaika GPS-datasta: {total_time_s:.1f} s")

# --------------------------------------------------
# KUVAAJAT
# --------------------------------------------------

st.subheader("Suodatettu kiihtyvyysdata")

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(time_acc, acc_filtered, label="Suodatettu kiihtyvyys")
ax1.plot(time_acc[peaks], acc_filtered[peaks], "x", label="Askelen huiput")
ax1.set_xlabel("Aika [s]")
ax1.set_ylabel("Kiihtyvyys [m/s²]")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

st.subheader("Tehospektritiheys (Welch)")

if freqs is not None and psd is not None:
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.semilogy(freqs, psd)
    ax2.set_xlim(0, 10)
    ax2.set_xlabel("Taajuus [Hz]")
    ax2.set_ylabel("PSD")
    if dominant_freq is not None:
        ax2.axvline(dominant_freq, color="r", linestyle="--", label=f"Dom. taajuus ≈ {dominant_freq:.2f} Hz")
        ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
else:
    st.info("FFT/PSD ei käytettävissä – liian vähän dataa?")

# --------------------------------------------------
# REITTI KARTALLA 
# --------------------------------------------------
st.subheader("Reittisi kartalla")
lat_col = next((c for c in df_loc.columns if "lat" in c.lower()), None)
lon_col = next((c for c in df_loc.columns if "lon" in c.lower() or "lng" in c.lower()), None)

if lat_col is not None and lon_col is not None:
    route_df = pd.DataFrame({
        "lat": pd.to_numeric(df_loc[lat_col], errors="coerce"),
        "lon": pd.to_numeric(df_loc[lon_col], errors="coerce")
    }).dropna()

    if len(route_df) > 0:
        # keskitä kartta reitin keskelle
        center = [float(route_df["lat"].mean()), float(route_df["lon"].mean())]
        m = folium.Map(location=center, zoom_start=15)

        # piirrä reitti
        folium.PolyLine(route_df[["lat", "lon"]].values.tolist(),
                        color="blue", weight=4, opacity=0.7).add_to(m)

        # start / end -markerit
        folium.Marker(route_df[["lat", "lon"]].iloc[0].tolist(),
                      popup="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(route_df[["lat", "lon"]].iloc[-1].tolist(),
                      popup="End", icon=folium.Icon(color="red")).add_to(m)

        # näytä folium-kartta streamlitissä
        st_folium(m, width=700, height=450)
    else:
        st.info("GPS-sarakkeet löytyivät, mutta niissä ei ole numeerista dataa.")
else:
    st.info("GPS-datasta ei löytynyt latitude/longitude -sarakkeita.")

# --------------------------------------------------
# TULKINTA
# --------------------------------------------------

st.subheader("Tulkinta")
st.write(f"""
**Mitatut tulokset:**
- **Askelmäärä (huiput):** {step_count_time} askelta
- **Askelmäärä (FFT):** {step_count_fft:.1f} askelta (arvio)
- **Kuljettu matka:** {distance_m/1000:.3f} km = {distance_m:.1f} m
- **Keskinopeus:** {avg_speed_m_s:.2f} m/s
- **Askelpituus:** {step_length_m:.2f} m

**Arviointi:**
- Keskimääräinen ihmisen askelpituus on 0,6-0,8 m, sinulla {step_length_m:.2f} m
- Keskimääräinen kävelynopeus on 1,4 m/s, sinulla {avg_speed_m_s:.2f} m/s
""")

st.subheader("FFT-analyysi komponenteittain")

fft_results = {}
for comp_name, comp_data in acc_components.items():
    if comp_data is not None:
        acc_filt = butter_bandpass_filter(comp_data, lowcut, highcut, fs)
        step_count_fft, _, _, dominant_freq = compute_steps_fft(
            acc_filt, fs, fmin=min_step_freq, fmax=max_step_freq
        )
        fft_results[comp_name] = {
            "steps": step_count_fft,
            "freq": dominant_freq
        }

col1, col2, col3 = st.columns(3)
for (comp_name, result), col in zip(fft_results.items(), [col1, col2, col3]):
    with col:
        if not np.isnan(result["steps"]):
            st.metric(
                f"{comp_name} FFT-askelmäärä",
                f"{result['steps']:.1f}",
                f"Taajuus: {result['freq']:.2f} Hz" if result["freq"] else "–"
            )

# --------------------------------------------------
# ANALYYSI: KIIHTYVYYDEN KOMPONENTIN VERTAILU
# --------------------------------------------------

# ❌ POISTA tämä rivi (se on jo määritelty ylempää):
# acc_components = { ... }

# Laske askelmäärä jokaiselle komponentille
peak_counts = {}
for comp_name, comp_data in acc_components.items():
    if comp_data is not None:
        acc_filt = butter_bandpass_filter(comp_data, lowcut, highcut, fs)
        peak_height = np.mean(acc_filt) + user_peak_height * np.std(acc_filt)
        peaks = signal.find_peaks(acc_filt, height=peak_height, distance=int(fs/4))[0]
        peak_counts[comp_name] = len(peaks)

# Näytä vertailu
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("X-komponentin askelmäärä", peak_counts.get("X (m/s^2)", "–"))
with col2:
    st.metric("Y-komponentin askelmäärä", peak_counts.get("Y (m/s^2)", "–"))
with col3:
    st.metric("Z-komponentin askelmäärä", peak_counts.get("Z (m/s^2)", "–"))

# Valitse paras komponentti
best_component = max(peak_counts, key=peak_counts.get)
st.info(f"✅ Paras komponentti: **{best_component}** ({peak_counts[best_component]} askelta)")