import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
sns.set(style="whitegrid")

# ============================
# 1. SETTING HALAMAN
# ============================
st.set_page_config(
    page_title="RFM + KMeans + Regresi HNI-HPAI",
    layout="wide"
)

st.title("📊 Analisis Penjualan 12 Bulan — RFM + KMeans + Regresi (Versi Cepat per-Step)")


# ============================
# TAB NAVIGASI
# ============================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["1️⃣ Upload Data", 
     "2️⃣ Preprocessing", 
     "3️⃣ RFM", 
     "4️⃣ KMeans", 
     "5️⃣ Regresi", 
     "6️⃣ Visualisasi"]
)

# ============================================================
# 1️⃣ TAB UPLOAD DATA
# ============================================================
with tab1:
    st.header("Upload File Excel")
    uploaded = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

    if uploaded:
        st.session_state["df_raw"] = pd.read_excel(uploaded)
        st.success("File berhasil diupload!")
        st.dataframe(st.session_state["df_raw"].head())
    else:
        st.info("Silakan upload file terlebih dahulu.")


# ============================================================
# 2️⃣ TAB PREPROCESSING
# ============================================================
with tab2:
    st.header("Preprocessing (Sama Persis Dengan Colab Anda)")

    if "df_raw" not in st.session_state:
        st.warning("Upload file terlebih dahulu di Tab 1.")
        st.stop()

    if st.button("🔧 Jalankan Preprocessing"):
        df = st.session_state["df_raw"].copy()

        # ------- CONVERSION ------
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

        # ------- MISSING VALUE ------
        for col in df.select_dtypes(include=['float','int']).columns:
            df[col].fillna(df[col].median(), inplace=True)
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

        # ------- FEATURE ENGINEERING ------
        df["Tgl. Penjualan"] = pd.to_datetime(df["Tgl. Penjualan"], errors="coerce")

        month_map = {
            1:"Januari",2:"Februari",3:"Maret",4:"April",5:"Mei",6:"Juni",
            7:"Juli",8:"Agustus",9:"September",10:"Oktober",11:"November",12:"Desember"
        }
        df["BulanNama"] = df["Tgl. Penjualan"].dt.month.map(month_map)
        df["HariPenjualan"] = df["Tgl. Penjualan"].dt.day_name()
        df["VP_per_Transaksi"] = df["Total VP"] / (df["Total"] + 1e-9)
        df["Panjang_Nama"] = df["Nama Pembeli"].apply(len)

        # ------- NORMALIZATION ------
        scaler_norm = MinMaxScaler()
        num_cols = ["Total", "Total VP", "VP_per_Transaksi"]
        for col in num_cols:
            if col in df.columns:
                df[col + "_Norm"] = scaler_norm.fit_transform(df[[col]])

        st.session_state["df_prep"] = df
        st.success("Preprocessing selesai!")
        st.dataframe(df.head(20))


# ============================================================
# 3️⃣ TAB RFM
# ============================================================
with tab3:
    st.header("RFM Analysis")

    if "df_prep" not in st.session_state:
        st.warning("Jalankan preprocessing dahulu.")
        st.stop()

    if st.button("📊 Jalankan RFM"):
        df = st.session_state["df_prep"]

        today = df["Tgl. Penjualan"].max() + pd.Timedelta(days=1)

        rfm = df.groupby(["ID Pembeli","Nama Pembeli"]).agg(
            Recency=("Tgl. Penjualan", lambda x: (today - x.max()).days),
            Frequency=("No. Invoice", "count"),
            Monetary=("Total", "sum"),
            Total_VP=("Total VP", "sum")
        ).reset_index()

        # RULE RFM (SAMA KODE COLAB)
        mon_q75 = rfm["Monetary"].quantile(0.75)
        rec_q50 = rfm["Recency"].median()
        freq_q50 = rfm["Frequency"].median()

        def segment(row):
            if row["Frequency"] >= freq_q50 and row["Monetary"] >= mon_q75:
                return "Loyalist / High Value"
            if row["Frequency"] <= 1 and row["Monetary"] >= mon_q75:
                return "Big Spender"
            if row["Recency"] > rec_q50 and row["Frequency"] <= freq_q50:
                return "At Risk"
            if row["Frequency"] <= 1:
                return "Occasional"
            return "Regular"

        rfm["Segment"] = rfm.apply(segment, axis=1)

        st.session_state["rfm"] = rfm
        st.success("RFM selesai!")
        st.dataframe(rfm.head(20))


# ============================================================
# 4️⃣ TAB KMEANS
# ============================================================
with tab4:
    st.header("KMeans Clustering")

    if "rfm" not in st.session_state:
        st.warning("Jalankan RFM dahulu.")
        st.stop()

    if st.button("🔍 Jalankan KMeans"):
        rfm = st.session_state["rfm"]

        X = rfm[["Recency","Frequency","Monetary","Total_VP"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Cari K Optimal
        scores = {}
        for k in range(2, 7):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            scores[k] = silhouette_score(X_scaled, labels)

        best_k = max(scores, key=scores.get)

        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        rfm["Cluster"] = km.fit_predict(X_scaled)

        centroids = scaler.inverse_transform(km.cluster_centers_)
        centroids_df = pd.DataFrame(
            centroids, 
            columns=["Recency","Frequency","Monetary","Total_VP"]
        )
        centroids_df["Cluster"] = centroids_df.index

        st.session_state["rfm_kmeans"] = rfm
        st.session_state["centroids"] = centroids_df

        st.success(f"KMeans selesai! k optimal = {best_k}")
        st.dataframe(centroids_df)


# ============================================================
# 5️⃣ TAB REGRESI
# ============================================================
with tab5:
    st.header("Regresi Linear")

    if "rfm_kmeans" not in st.session_state:
        st.warning("Jalankan KMeans dahulu.")
        st.stop()

    if st.button("📈 Jalankan Regresi"):
        rfm_k = st.session_state["rfm_kmeans"]

        X = rfm_k[["Recency","Frequency","Monetary","Total_VP"]]
        y = rfm_k["Monetary"]

        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)

        poly = PolynomialFeatures(2, include_bias=False)
        X_poly = poly.fit_transform(X_scaled)

        X_train, X_test, y_train, y_test = train_test_split(
            X_poly, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        st.write("R² (Test):", r2_score(y_test, pred))
        st.write("MSE (Test):", mean_squared_error(y_test, pred))

        st.session_state["reg_pred"] = (y_test, pred)

        st.success("Regresi selesai!")


# ============================================================
# 6️⃣ TAB VISUALISASI
# ============================================================
with tab6:
    st.header("Visualisasi Grafik")

    if "rfm_kmeans" not in st.session_state:
        st.warning("Jalankan semua step terlebih dahulu.")
        st.stop()

    rfm = st.session_state["rfm_kmeans"]
    centroids_df = st.session_state["centroids"]

    # Scatter Clustering
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=rfm, x="Frequency", y="Monetary", hue="Cluster", palette="tab10")
    plt.title("Clustering (Frequency vs Monetary)")
    st.pyplot(fig)

    # Scatter + Centroid
    fig2, ax2 = plt.subplots(figsize=(8,6))
    plt.scatter(rfm["Frequency"], rfm["Monetary"], c=rfm["Cluster"], cmap="tab10")
    plt.scatter(
        centroids_df["Frequency"],
        centroids_df["Monetary"],
        c="red", s=200, marker="X"
    )
    plt.title("Cluster Centroids")
    st.pyplot(fig2)

    # Regresi
    if "reg_pred" in st.session_state:
        y_test, pred = st.session_state["reg_pred"]

        fig3, ax3 = plt.subplots(figsize=(7,5))
        ax3.scatter(y_test, pred)
        ax3.plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], "r--")
        ax3.set_title("Regresi: Aktual vs Prediksi")
        st.pyplot(fig3)
