import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # WAJIB AGAR STREAMLIT TIDAK ERROR
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid")

SAMPLE_PATH = "Penjualan_Sintetik_12bulan_V3.xlsx"

# 1. LOAD DATA
def load_data(uploaded_file=None, use_sample=False):
    if uploaded_file:
        return pd.read_excel(uploaded_file)
    if use_sample:
        return pd.read_excel(SAMPLE_PATH)
    return None

# 2. PREPROCESSING
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
            try:
                df[col] = df[col].astype(dtype)
            except:
                pass

    # Missing values
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

    # Normalisasi MinMax
    num_cols = ["Total", "Total VP", "VP_per_Transaksi"]
    scaler_norm = MinMaxScaler()
    df[[col+"_Norm" for col in num_cols]] = scaler_norm.fit_transform(df[num_cols])

    return df

# 3. RFM
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

# 4. KMEANS
def run_kmeans(rfm, min_k=2, max_k=7):
    X = rfm[["Recency","Frequency","Monetary","Total_VP"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_scores = {}
    for k in range(min_k, max_k+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        silhouette_scores[k] = silhouette_score(X_scaled, labels)

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(X_scaled)

    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids_original, columns=X.columns)
    centroids_df["Cluster"] = range(best_k)

    return rfm, centroids_df, silhouette_scores, rfm["Cluster"].value_counts().sort_index(), best_k

# 5. REGRESI
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
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2, y_test, y_pred


# 6. STREAMLIT UI

st.title(" Analisis Penjualan 12 Bulan — RFM + KMeans + Regresi")
st.write("Upload file atau gunakan data sampel untuk memulai analisis.")

uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx","xls"])
use_sample = st.checkbox("Gunakan sample dataset", value=True)

df_load = load_data(uploaded_file=uploaded_file, use_sample=use_sample)

if df_load is not None:
    st.subheader("Preview Data Awal")
    st.dataframe(df_load.head())

    if st.button("Jalankan Analisis Lengkap"):
        df = preprocess(df_load)
        rfm = compute_rfm(df)
        rfm_k, centroids, silh, counts, best_k = run_kmeans(rfm)
        mse, r2, y_test, y_pred = run_regression(rfm_k)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Preprocessing", 
            "RFM", 
            "K-Means",
            "Regresi",
            "Visualisasi"
        ])

        # TAB 1
        with tab1:
            st.subheader("Hasil Preprocessing")
            st.dataframe(df.head())

        # TAB 2
        with tab2:
            st.subheader("Hasil RFM")
            st.dataframe(rfm)

        # TAB 3
        with tab3:
            st.subheader("Silhouette Score")
            st.table(pd.DataFrame(silh.items(), columns=["k","Silhouette Score"]))

            st.subheader(f"Cluster Optimal = {best_k}")
            st.dataframe(centroids)

            st.subheader("Jumlah Anggota per Cluster")
            st.table(counts)

        # TAB 4 
        with tab4:
            st.write(f"MSE Regresi: **{mse:.2f}**")
            st.write(f"R² Regresi: **{r2:.4f}**")

        # TAB 5
        with tab5:
            st.subheader("Scatter Cluster RFM")
            fig, ax = plt.subplots(figsize=(6,5))
            sns.scatterplot(data=rfm_k, x="Frequency", y="Monetary", hue="Cluster", palette="tab10", s=60)
            st.pyplot(fig)

            st.subheader("Heatmap Korelasi RFM")
            fig2, ax2 = plt.subplots(figsize=(6,5))
            sns.heatmap(rfm_k[["Recency","Frequency","Monetary","Total_VP"]].corr(), annot=True, cmap="coolwarm")
            st.pyplot(fig2)

else:
    st.info("Silakan upload file atau gunakan sample dataset.")
