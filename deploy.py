# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # penting untuk lingkungan deploy
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

# -----------------------
# UTIL FUNCTIONS
# -----------------------
def load_data(uploaded_file=None, use_sample=False):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)
    if use_sample:
        try:
            return pd.read_excel(SAMPLE_PATH)
        except Exception as e:
            st.error(f"Gagal memuat file sample: {e}")
            return None
    return None

def preprocess(df):
    df = df.copy()
    # conversions
    conv = {
        "No. Invoice": "str",
        "ID Pembeli": "str",
        "Nama Pembeli": "str"
    }
    for c, t in conv.items():
        if c in df.columns:
            try:
                df[c] = df[c].astype(t)
            except Exception:
                pass

    # numeric ensure
    for col in ["Total", "Total VP"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # handle missing
    # numeric -> median, object -> mode (safely)
    for col in df.select_dtypes(include=["number"]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().any():
            try:
                df[col].fillna(df[col].mode().iloc[0], inplace=True)
            except Exception:
                df[col].fillna("Unknown", inplace=True)

    # datetime
    if "Tgl. Penjualan" in df.columns:
        df["Tgl. Penjualan"] = pd.to_datetime(df["Tgl. Penjualan"], errors="coerce")
    else:
        st.error("Kolom 'Tgl. Penjualan' tidak ditemukan di dataset.")
        return df

    # features
    month_map = {1:"Januari",2:"Februari",3:"Maret",4:"April",5:"Mei",6:"Juni",
                 7:"Juli",8:"Agustus",9:"September",10:"Oktober",11:"November",12:"Desember"}
    df["BulanNama"] = df["Tgl. Penjualan"].dt.month.map(month_map)
    df["HariPenjualan"] = df["Tgl. Penjualan"].dt.day_name()
    df["VP_per_Transaksi"] = df["Total VP"] / (df["Total"].replace(0, np.nan))
    df["VP_per_Transaksi"].fillna(0, inplace=True)
    df["Panjang_Nama"] = df["Nama Pembeli"].astype(str).apply(len)

    # Normalization (min-max) - optional columns
    num_cols = [c for c in ["Total", "Total VP", "VP_per_Transaksi"] if c in df.columns]
    if num_cols:
        scaler_norm = MinMaxScaler()
        df[[c + "_Norm" for c in num_cols]] = scaler_norm.fit_transform(df[num_cols])

    return df

def compute_rfm(df):
    today = df["Tgl. Penjualan"].max() + pd.Timedelta(days=1)
    rfm = df.groupby(["ID Pembeli","Nama Pembeli"]).agg(
        Recency = ("Tgl. Penjualan", lambda x: (today - x.max()).days),
        Frequency = ("No. Invoice", "count"),
        Monetary = ("Total", "sum"),
        Total_VP = ("Total VP", "sum")
    ).reset_index()

    # segmentation (deterministic)
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

def run_kmeans(rfm, min_k=2, max_k=7, random_state=42):
    X = rfm[["Recency","Frequency","Monetary","Total_VP"]].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_scores = {}
    for k in range(min_k, max_k+1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        # calculate silhouette only if feasible
        try:
            score = silhouette_score(X_scaled, labels)
        except Exception:
            score = -1
        silhouette_scores[k] = score

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    rfm["Cluster"] = labels

    centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids_orig, columns=["Recency","Frequency","Monetary","Total_VP"])
    centroids_df["Cluster"] = centroids_df.index

    cluster_counts = rfm["Cluster"].value_counts().sort_index()
    return rfm, centroids_df, silhouette_scores, cluster_counts, best_k

def run_regression(rfm, degree=2, test_size=0.2, random_state=42, add_noise=False, noise_factor=0.05):
    # Default: do NOT add noise to keep deterministic and consistent
    X_reg = rfm[["Recency","Frequency","Monetary","Total_VP"]].copy()
    y = rfm["Monetary"].copy()

    scaler_reg = StandardScaler()
    X_scaled = scaler_reg.fit_transform(X_reg)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=test_size, random_state=random_state)

    if add_noise:
        np.random.seed(random_state)
        y_train = y_train + np.random.normal(0, y_train.std() * noise_factor, size=len(y_train))

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    return {"model": model, "poly": poly, "scaler_reg": scaler_reg,
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "y_pred_train": y_pred_train, "y_pred_test": y_pred_test,
            "mse_train": mse_train, "mse_test": mse_test, "r2_train": r2_train, "r2_test": r2_test}

def to_excel_bytes(dfs_map):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name, df in dfs_map.items():
            sheet = name.replace(".xlsx","")[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
    out.seek(0)
    return out

# -----------------------
# STREAMLIT UI (Tab-based)
# -----------------------
st.set_page_config(page_title="Analisis RFM + KMeans + Regresi", layout="wide")
st.title("Analisis Penjualan 12 Bulan — RFM + KMeans + Regresi")

# Sidebar
with st.sidebar:
    st.header("Input")
    uploaded_file = st.file_uploader("Upload file Excel (.xlsx/.xls)", type=["xlsx","xls"])
    use_sample = st.checkbox("Gunakan sample file (harus ada di repo/host)", value=(uploaded_file is None))
    run_pipeline = st.button("Jalankan Pipeline")
    st.markdown("---")
    st.write("Sample path (edit jika diperlukan):")
    st.code(SAMPLE_PATH)

# Load
df_loaded = load_data(uploaded_file=uploaded_file, use_sample=use_sample)
if df_loaded is None:
    st.info("Silakan upload file atau centang 'Gunakan sample file' (dan pastikan file ada di path).")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Preprocessing", "RFM", "Clustering", "Regresi & Download"])

with tab1:
    st.subheader("Preview Data Awal")
    st.dataframe(df_loaded.head(10))
    st.write("Ukuran dataset:", df_loaded.shape)
    st.write("Cek missing values:")
    st.table(df_loaded.isnull().sum().rename("Missing_Count"))

# Run pipeline when requested
if run_pipeline:
    try:
        # Preprocess
        df = preprocess(df_loaded)
        with tab2:
            st.subheader("Hasil Preprocessing (preview)")
            st.dataframe(df.head(20))
            st.write("Tipe kolom:")
            st.table(pd.DataFrame(df.dtypes, columns=["dtype"]))

        # RFM
        rfm = compute_rfm(df)
        with tab3:
            st.subheader("RFM (semua pelanggan)")
            # Tampilkan semua RFM (hati-hati besar)
            st.write("Jumlah pelanggan (unique):", rfm.shape[0])
            st.dataframe(rfm.sort_values(["Monetary"], ascending=False).reset_index(drop=True))
            st.download_button("Unduh RFM (Excel)", data=to_excel_bytes({"RFM.xlsx": rfm}), file_name="RFM.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # KMeans
        rfm_k, centroids_df, silhouette_scores, cluster_counts, best_k = run_kmeans(rfm)
        with tab4:
            st.subheader("K-Means")
            st.write(f"Cluster optimal (berdasarkan Silhouette): {best_k}")
            st.table(pd.DataFrame(list(silhouette_scores.items()), columns=["k","Silhouette"]))
            st.subheader("Centroid (skala asli)")
            st.dataframe(centroids_df)
            st.subheader("Jumlah anggota per cluster")
            st.table(cluster_counts.reset_index().rename(columns={"index":"Cluster", "Cluster":"Count"}).set_index("Cluster"))
            st.subheader("Scatter Frequency vs Monetary (colored by Cluster)")
            fig, ax = plt.subplots(figsize=(8,5))
            sns.scatterplot(data=rfm_k, x="Frequency", y="Monetary", hue="Cluster", palette="tab10", ax=ax, s=60)
            for i, row in centroids_df.iterrows():
                ax.scatter(row["Frequency"], row["Monetary"], c="red", marker="X", s=150)
                ax.text(row["Frequency"], row["Monetary"], f"C{int(row['Cluster'])}", fontsize=10, fontweight="bold")
            st.pyplot(fig)
            plt.clf()

        # Regression
        reg_info = run_regression(rfm_k, add_noise=False)
        with tab5:
            st.subheader("Regresi Linear (validasi)")
            st.write(f"Train MSE: **{reg_info['mse_train']:.2f}**, Test MSE: **{reg_info['mse_test']:.2f}**")
            st.write(f"Train R²: **{reg_info['r2_train']:.4f}**, Test R²: **{reg_info['r2_test']:.4f}**")
            # plot actual vs pred (test)
            if len(reg_info["y_test"]) > 0:
                fig2, ax2 = plt.subplots(figsize=(7,5))
                ax2.scatter(reg_info["y_test"], reg_info["y_pred_test"], color="steelblue", alpha=0.7)
                ax2.plot([reg_info["y_test"].min(), reg_info["y_test"].max()],
                         [reg_info["y_test"].min(), reg_info["y_test"].max()], "r--")
                ax2.set_xlabel("Actual Monetary")
                ax2.set_ylabel("Predicted Monetary")
                st.pyplot(fig2)
                plt.clf()

            # Save / download all results
            dfs = {"RFM.xlsx": rfm_k, "Centroid.xlsx": centroids_df}
            excel_bytes = to_excel_bytes(dfs)
            st.download_button("Unduh Semua Hasil (Excel)", data=excel_bytes,
                                file_name="Hasil_RFM_KMeans_Regresi.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.success("Pipeline selesai tanpa error.")
    except Exception as e:
        st.exception(f"Terjadi error saat menjalankan pipeline: {e}")
else:
    st.info("Tekan tombol 'Jalankan Pipeline' pada sidebar untuk memproses data.")
