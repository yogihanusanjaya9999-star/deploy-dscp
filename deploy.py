import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO

# ML
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# ====== PATH SAMPLE DATA (jika ingin pakai file default) ======
SAMPLE_PATH = "Penjualan_Sintetik_12bulan_V3.xlsx"


# =================================================================
# 1. LOAD DATA
# =================================================================
def load_data(uploaded_file=None, use_sample=False):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)
    if use_sample:
        return pd.read_excel(SAMPLE_PATH)
    return None


# =================================================================
# 2. PREPROCESSING (SAMA SEPERTI KODE COLAB)
# =================================================================
def preprocess(df):
    df = df.copy()

    # Konversi tipe data
    conversion_map = {
        "No. Invoice": "str",
        "ID Pembeli": "str",
        "Nama Pembeli": "str",
        "Total": "float",
        "Total VP": "float"
    }
    for col, dtype in conversion_map.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except:
                pass

    # Handling Missing Value
    for col in df.select_dtypes(include=['float','int']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Feature Engineering
    df["Tgl. Penjualan"] = pd.to_datetime(df["Tgl. Penjualan"], errors="coerce")

    month_map = {
        1:"Januari",2:"Februari",3:"Maret",4:"April",5:"Mei",6:"Juni",
        7:"Juli",8:"Agustus",9:"September",10:"Oktober",11:"November",12:"Desember"
    }

    df["BulanNama"] = df["Tgl. Penjualan"].dt.month.map(month_map)
    df["VP_per_Transaksi"] = df["Total VP"] / (df["Total"] + 1e-9)
    df["Panjang_Nama"] = df["Nama Pembeli"].astype(str).apply(len)

    # Normalisasi
    scaler = MinMaxScaler()
    numeric_cols = ["Total", "Total VP", "VP_per_Transaksi"]
    df[[c+"_Norm" for c in numeric_cols]] = scaler.fit_transform(df[numeric_cols])

    return df


# =================================================================
# 3. RFM ANALYSIS
# =================================================================
def compute_rfm(df):
    today = df["Tgl. Penjualan"].max() + pd.Timedelta(days=1)

    rfm = df.groupby(["ID Pembeli", "Nama Pembeli"]).agg(
        Recency=("Tgl. Penjualan", lambda x: (today - x.max()).days),
        Frequency=("No. Invoice", "count"),
        Monetary=("Total", "sum"),
        Total_VP=("Total VP", "sum")
    ).reset_index()

    # RFM thresholds
    mon_q75 = rfm["Monetary"].quantile(0.75)
    freq_q50 = rfm["Frequency"].median()
    rec_q50 = rfm["Recency"].median()

    def rfm_segment(row):
        if row["Frequency"] >= freq_q50 and row["Monetary"] >= mon_q75:
            return "Loyalist / High Value"
        if row["Frequency"] <= 1 and row["Monetary"] >= mon_q75:
            return "Big Spender"
        if row["Recency"] > rec_q50 and row["Frequency"] <= freq_q50:
            return "At Risk"
        if row["Frequency"] <= 1:
            return "Occasional"
        return "Regular"

    rfm["Segment"] = rfm.apply(rfm_segment, axis=1)
    return rfm


# =================================================================
# 4. K-MEANS
# =================================================================
def run_kmeans(rfm):
    X = rfm[["Recency", "Frequency", "Monetary", "Total_VP"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cari best_k
    silhouette_scores = {}
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        silhouette_scores[k] = silhouette_score(X_scaled, labels)

    best_k = max(silhouette_scores, key=silhouette_scores.get)

    # Final KMeans
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm["Cluster"] = model.fit_predict(X_scaled)

    centroids_scaled = model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    centroids_df = pd.DataFrame(
        centroids_original,
        columns=["Recency", "Frequency", "Monetary", "Total_VP"]
    )
    centroids_df["Cluster"] = range(best_k)

    return rfm, centroids_df, silhouette_scores, best_k


# =================================================================
# 5. REGRESI LINIER
# =================================================================
def run_regression(rfm):
    X = rfm[["Recency", "Frequency", "Monetary", "Total_VP"]]
    y = rfm["Monetary"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )

    # noise kecil seperti kode colab
    y_train_noisy = y_train + np.random.normal(0, y_train.std()*0.05, len(y_train))

    model = LinearRegression()
    model.fit(X_train, y_train_noisy)

    y_pred_test = model.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)

    return r2_test, mse_test


# =================================================================
# 6. ALTair helper
# =================================================================
def plot_cluster_altair(rfm, centroids):
    scatter = alt.Chart(rfm).mark_circle(size=60).encode(
        x="Frequency:Q",
        y="Monetary:Q",
        color="Cluster:N",
        tooltip=["Nama Pembeli", "Frequency", "Monetary", "Cluster"]
    )

    centroid_chart = alt.Chart(centroids).mark_point(
        size=200, shape="cross", color="red"
    ).encode(
        x="Frequency:Q",
        y="Monetary:Q",
        tooltip=["Cluster", "Frequency", "Monetary"]
    )

    return (scatter + centroid_chart).interactive()


# =================================================================
# 7. STREAMLIT UI
# =================================================================
st.title("📊 Analisis Penjualan 12 Bulan — RFM + KMeans + Regresi (Versi Super Ringan)")

uploaded = st.file_uploader("Upload file Excel", type=["xlsx"])
use_sample = st.checkbox("Gunakan file sample", value=(uploaded is None))

if uploaded is None and not use_sample:
    st.stop()

df = load_data(uploaded_file=uploaded, use_sample=use_sample)
st.success("Data berhasil di-load!")
st.dataframe(df.head())

if st.button("Jalankan Proses Analisis"):
    # PREPROCESS
    df_prep = preprocess(df)
    st.subheader("Hasil Preprocessing")
    st.dataframe(df_prep.head())

    # RFM
    rfm = compute_rfm(df_prep)
    st.subheader("Hasil RFM")
    st.dataframe(rfm.head())

    # KMeans
    rfm_k, centroids_df, scores, best_k = run_kmeans(rfm)
    st.success(f"K-Means selesai! Cluster optimal = {best_k}")

    st.subheader("Silhouette Score")
    st.write(scores)

    st.subheader("Centroid")
    st.dataframe(centroids_df)

    # ALTair Visual
    st.subheader("📌 Visualisasi Cluster (Altair — super cepat)")
    chart = plot_cluster_altair(rfm_k, centroids_df)
    st.altair_chart(chart, use_container_width=True)

    # REGRESI
    r2, mse = run_regression(rfm_k)
    st.subheader("📈 Hasil Regresi Linier")
    st.write(f"R² Test = **{r2:.4f}**")
    st.write(f"MSE Test = **{mse:.2f}**")

    st.success("Pipeline selesai tanpa visualisasi berat 💨")
