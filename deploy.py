import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # WAJIB UNTUK DEPLOY STREAMLIT
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid")

# ==========================
# KONFIGURASI FILE SAMPLE
# ==========================
SAMPLE_PATH = "Penjualan_Sintetik_12bulan_V3.xlsx"


# ==========================
# LOAD DATA
# ==========================
def load_data(uploaded_file=None, use_sample=False):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)
    if use_sample:
        return pd.read_excel(SAMPLE_PATH)
    return None


# ==========================
# PREPROCESSING
# ==========================
def preprocess(df):
    df = df.copy()

    conversion_map = {
        "No. Invoice": "str",
        "ID Pembeli": "str",
        "Nama Pembeli": "str",
        "Total": "float",
        "Total VP": "float"
    }
    for col, dtype in conversion_map.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype, errors="ignore")

    # Handling missing
    for col in df.select_dtypes(include=['float','int']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Feature engineering
    df["Tgl. Penjualan"] = pd.to_datetime(df["Tgl. Penjualan"], errors="coerce")

    month_map = {
        1:"Januari",2:"Februari",3:"Maret",4:"April",5:"Mei",6:"Juni",
        7:"Juli",8:"Agustus",9:"September",10:"Oktober",11:"November",12:"Desember"
    }
    df["BulanNama"] = df["Tgl. Penjualan"].dt.month.map(month_map)
    df["HariPenjualan"] = df["Tgl. Penjualan"].dt.day_name()

    df["VP_per_Transaksi"] = df["Total VP"] / (df["Total"] + 1e-9)
    df["Panjang_Nama"] = df["Nama Pembeli"].astype(str).apply(len)

    scaler_norm = MinMaxScaler()
    cols = ["Total","Total VP","VP_per_Transaksi"]
    df[[c+"_Norm" for c in cols]] = scaler_norm.fit_transform(df[cols])

    return df


# ==========================
# RFM COMPUTATION
# ==========================
def compute_rfm(df):

    today = df["Tgl. Penjualan"].max() + pd.Timedelta(days=1)

    rfm = df.groupby(["ID Pembeli","Nama Pembeli"]).agg(
        Recency=("Tgl. Penjualan", lambda x: (today - x.max()).days),
        Frequency=("No. Invoice", "count"),
        Monetary=("Total", "sum"),
        Total_VP=("Total VP", "sum")
    ).reset_index()

    mon_q75 = rfm["Monetary"].quantile(0.75)
    rec_q50 = rfm["Recency"].median()
    freq_q50 = rfm["Frequency"].median()

    def seg(row):
        if row["Frequency"] >= freq_q50 and row["Monetary"] >= mon_q75:
            return "Loyalist / High Value"
        if row["Frequency"] <= 1 and row["Monetary"] >= mon_q75:
            return "Big Spender"
        if row["Recency"] > rec_q50 and row["Frequency"] <= freq_q50:
            return "At Risk"
        if row["Frequency"] <= 1:
            return "Occasional"
        return "Regular"

    rfm["Segment"] = rfm.apply(seg, axis=1)
    return rfm


# ==========================
# K-MEANS
# ==========================
def run_kmeans(rfm, min_k=2, max_k=7):
    X = rfm[["Recency","Frequency","Monetary","Total_VP"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_scores = {}
    for k in range(min_k, max_k+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        try:
            silhouette_scores[k] = silhouette_score(X_scaled, labels)
        except:
            silhouette_scores[k] = -1

    best_k = max(silhouette_scores, key=silhouette_scores.get)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm["Cluster"] = km.fit_predict(X_scaled)

    centroids_original = scaler.inverse_transform(km.cluster_centers_)
    centroids_df = pd.DataFrame(centroids_original, columns=X.columns)
    centroids_df["Cluster"] = range(best_k)

    return rfm, centroids_df, silhouette_scores, rfm["Cluster"].value_counts(), best_k


# ==========================
# REGRESSION
# ==========================
def run_regression(rfm):

    X = rfm[["Recency","Frequency","Monetary","Total_VP"]]
    y = rfm["Monetary"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, y_test, y_pred


# ==========================
# EXCEL EXPORT
# ==========================
def to_excel_bytes(dfs):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    buffer.seek(0)
    return buffer


# ==========================
# STREAMLIT UI
# ==========================

st.title("Analisis Penjualan 12 Bulan — HNI HPAI")

st.sidebar.header("Input Data")
uploaded = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])
sample = st.sidebar.checkbox("Gunakan Sample", value=(uploaded is None))
run = st.sidebar.button("Jalankan Analisis")

df_raw = load_data(uploaded, sample)

if df_raw is not None:
    st.subheader("Preview Data Awal")
    st.dataframe(df_raw.head())

if run:

    df = preprocess(df_raw)
    st.success("Preprocessing selesai.")
    st.dataframe(df.head())

    rfm = compute_rfm(df)
    st.success("RFM selesai.")
    st.dataframe(rfm.head())

    rfm, centroids, scores, counts, bestk = run_kmeans(rfm)
    st.write("Cluster optimal:", bestk)
    st.dataframe(centroids)

    model, y_test, y_pred = run_regression(rfm)
    st.write("Regresi R²:", r2_score(y_test, y_pred))

    # EXPORT
    excel_bytes = to_excel_bytes({
        "RFM": rfm,
        "Centroid": centroids
    })

    st.download_button("Unduh Hasil Excel", excel_bytes, "hasil.xlsx")
