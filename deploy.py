# app.py
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

st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set(style="whitegrid")

# ---------- helper functions ----------
SAMPLE_PATH = "Penjualan_Sintetik_12bulan_V3.xlsx"  # gunakan path ini jika ingin sampel

def load_data(uploaded_file=None, use_sample=False):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        return df
    if use_sample:
        df = pd.read_excel(SAMPLE_PATH)
        return df
    return None

def preprocess(df):
    df = df.copy()
    # conversion
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
            except Exception:
                pass

    # handling missing
    for col in df.select_dtypes(include=['float','int']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode().iloc[0], inplace=True)

    # feature engineering
    df["Tgl. Penjualan"] = pd.to_datetime(df["Tgl. Penjualan"], errors="coerce")
    month_map = {
        1:"Januari",2:"Februari",3:"Maret",4:"April",5:"Mei",6:"Juni",
        7:"Juli",8:"Agustus",9:"September",10:"Oktober",11:"November",12:"Desember"
    }
    df["BulanNama"] = df["Tgl. Penjualan"].dt.month.map(month_map)
    df["HariPenjualan"] = df["Tgl. Penjualan"].dt.day_name()
    # derived features
    df["VP_per_Transaksi"] = df["Total VP"] / (df["Total"] + 1e-9)
    df["Panjang_Nama"] = df["Nama Pembeli"].astype(str).apply(len)

    # normalization columns (MinMax)
    scaler_norm = MinMaxScaler()
    num_cols = []
    for c in ["Total", "Total VP", "VP_per_Transaksi"]:
        if c in df.columns:
            num_cols.append(c)
    if num_cols:
        df[[c+"_Norm" for c in num_cols]] = scaler_norm.fit_transform(df[num_cols])
    return df

def compute_rfm(df):
    today = df["Tgl. Penjualan"].max() + pd.Timedelta(days=1)
    rfm = df.groupby(["ID Pembeli","Nama Pembeli"]).agg(
        Recency = ("Tgl. Penjualan", lambda x: (today - x.max()).days),
        Frequency = ("No. Invoice", "count"),
        Monetary = ("Total", "sum"),
        Total_VP = ("Total VP", "sum")
    ).reset_index()

    # rfm segmentation rules (mengikuti skrip asli)
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

def run_kmeans(rfm, min_k=2, max_k=7):
    X = rfm[["Recency", "Frequency", "Monetary", "Total_VP"]].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette_scores = {}
    for k in range(min_k, max_k+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        try:
            score = silhouette_score(X_scaled, labels)
        except Exception:
            score = -1
        silhouette_scores[k] = score

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    rfm["Cluster"] = labels

    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids_original, columns=["Recency", "Frequency", "Monetary", "Total_VP"])
    centroids_df["Cluster"] = range(best_k)

    cluster_counts = rfm["Cluster"].value_counts().sort_index()
    return rfm, centroids_df, silhouette_scores, cluster_counts, best_k

def run_regression(rfm, degree=2, test_size=0.2, noise_factor=0.05):
    X_reg = rfm[["Recency", "Frequency", "Monetary", "Total_VP"]].copy()
    y = rfm["Monetary"].copy()

    scaler_reg = StandardScaler()
    X_scaled = scaler_reg.fit_transform(X_reg)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=test_size, random_state=42)
    # tambahkan noise kecil ke y_train (seperti skrip asli)
    np.random.seed(42)
    y_train_noisy = y_train + np.random.normal(0, y_train.std() * noise_factor, size=len(y_train))

    model = LinearRegression()
    model.fit(X_train, y_train_noisy)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    return {
        "model": model,
        "poly": poly,
        "scaler_reg": scaler_reg,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred_train": y_pred_train, "y_pred_test": y_pred_test,
        "mse_train": mse_train, "mse_test": mse_test,
        "r2_train": r2_train, "r2_test": r2_test
    }

def to_excel_bytes(dfs_map):
    # dfs_map: {"filename.xlsx": dataframe}
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, df in dfs_map.items():
            safe_name = name.replace(".xlsx","")
            df.to_excel(writer, sheet_name=safe_name[:31], index=False)
    output.seek(0)
    return output

# ---------- Streamlit UI ----------
st.title("Analisis Penjualan 12 Bulan — HNI HPAI (RFM + KMeans + Regresi)")
st.markdown("Aplikasi deploy untuk pipeline analisis: preprocessing → RFM → K-Means → Regresi → visualisasi.\
            Upload file Excel atau gunakan sample file yang disediakan.")

with st.sidebar:
    st.header("Input & Opsi")
    uploaded_file = st.file_uploader("Upload file Excel (format seperti dataset Anda)", type=["xlsx","xls"])
    use_sample = st.checkbox("Gunakan file sampel internal", value=(uploaded_file is None))
    show_pairplot = st.checkbox("Tampilkan Pairplot RFM (membutuhkan waktu)", value=False)
    run_pipeline = st.button("Jalankan Analisis (Pipeline Lengkap)")
    st.markdown("---")
    st.markdown("File sample path (internal):")
    st.write(SAMPLE_PATH)

if uploaded_file is None and not use_sample:
    st.warning("Silakan upload file .xlsx atau pilih 'Gunakan file sampel internal' di sidebar.")
    st.stop()

# load data
df_load = load_data(uploaded_file=uploaded_file, use_sample=use_sample)
if df_load is None:
    st.error("Tidak dapat memuat data. Periksa kembali input.")
    st.stop()

st.subheader("Preview Data Awal (5 baris)")
st.dataframe(df_load.head())

if run_pipeline:
    try:
        with st.spinner("Proses preprocessing..."):
            df = preprocess(df_load)
        st.success("Preprocessing selesai.")
        st.subheader("Hasil Preprocessing (preview 10 baris)")
        st.dataframe(df.head(10))

        # RFM
        with st.spinner("Menghitung RFM..."):
            rfm = compute_rfm(df)
        st.success("RFM selesai.")
        st.subheader("Hasil RFM (preview 20 baris)")
        st.dataframe(rfm.head(20))

        # KMeans
        with st.spinner("Menentukan cluster optimal & menjalankan KMeans..."):
            rfm_k, centroids_df, silhouette_scores, cluster_counts, best_k = run_kmeans(rfm)
        st.success(f"KMeans selesai (k_optimal={best_k}).")

        st.subheader("Silhouette Score per k")
        st.table(pd.DataFrame.from_dict(silhouette_scores, orient="index", columns=["Silhouette"]).reset_index().rename(columns={"index":"k"}))

        st.subheader("Titik Centroid (skala asli)")
        st.dataframe(centroids_df)

        st.subheader("Jumlah anggota tiap cluster")
        st.table(cluster_counts.reset_index().rename(columns={"index":"Cluster", "Cluster":"Count"}).set_index("Cluster"))

        # Regresi
        with st.spinner("Menjalankan regresi linier (validasi)..."):
            reg_info = run_regression(rfm_k)
        st.success("Regresi selesai.")

        st.subheader("Hasil Regresi (Train/Test split 80:20)")
        st.write(f"Train MSE: **{reg_info['mse_train']:.2f}**")
        st.write(f"Test MSE : **{reg_info['mse_test']:.2f}**")
        st.write(f"Train R² : **{reg_info['r2_train']:.4f}**")
        st.write(f"Test R²  : **{reg_info['r2_test']:.4f}**")

        # Visualizations
        st.subheader("Visualisasi")
        # Distribution histograms
        fig1, ax = plt.subplots(1,1, figsize=(10,4))
        df[["Total","Total VP","VP_per_Transaksi"]].hist(bins=20, figsize=(14,6), color='skyblue', edgecolor='black')
        plt.suptitle("Distribusi Fitur Numerik (Setelah Preprocessing)", fontsize=14)
        st.pyplot(plt.gcf())
        plt.clf()

        # Boxplot
        fig2, ax = plt.subplots(figsize=(8,4))
        sns.boxplot(data=df[["Total","Total VP","VP_per_Transaksi"]], palette="Set2", ax=ax)
        plt.title("Boxplot: Deteksi Outlier Fitur Numerik")
        st.pyplot(fig2)
        plt.clf()

        # Pairplot (opsional)
        if show_pairplot:
            st.write("Menghasilkan pairplot — mohon tunggu...")
            try:
                pp = sns.pairplot(rfm_k[["Recency","Frequency","Monetary","Total_VP","Segment"]], hue="Segment", palette="husl", diag_kind="kde")
                st.pyplot(pp.fig)
                plt.clf()
            except Exception as e:
                st.error(f"Pairplot gagal dibuat: {e}")

        # Segment distribution
        fig3, ax = plt.subplots(figsize=(7,4))
        rfm_k["Segment"].value_counts().plot(kind="bar", color="cornflowerblue", edgecolor="black", ax=ax)
        ax.set_title("Distribusi Segmen RFM")
        ax.set_xlabel("Segmen")
        ax.set_ylabel("Jumlah Pelanggan")
        st.pyplot(fig3)
        plt.clf()

        # KMeans scatter (Frequency vs Monetary)
        fig4, ax = plt.subplots(figsize=(7,5))
        sns.scatterplot(data=rfm_k, x="Frequency", y="Monetary", hue="Cluster", palette="tab10", s=60, ax=ax)
        ax.set_title("K-Means Clustering: Frequency vs Monetary")
        st.pyplot(fig4)
        plt.clf()

        # Replot centroids overlay
        fig5, ax = plt.subplots(figsize=(8,6))
        scatter = ax.scatter(rfm_k["Frequency"], rfm_k["Monetary"], c=rfm_k["Cluster"], cmap="tab10", s=50, alpha=0.6)
        ax.scatter(centroids_df["Frequency"], centroids_df["Monetary"], c="red", s=200, marker="X", label="Centroid")
        for i, row in centroids_df.iterrows():
            ax.text(row["Frequency"], row["Monetary"], f"C{int(row['Cluster'])}", fontsize=10, fontweight="bold", color="black", ha="center")
        ax.set_title("Visualisasi Klaster dan Titik Centroid")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Monetary")
        ax.legend()
        st.pyplot(fig5)
        plt.clf()

        # Heatmap korelasi RFM
        fig6, ax = plt.subplots(figsize=(6,5))
        corr = rfm_k[["Recency","Frequency","Monetary","Total_VP"]].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Heatmap Korelasi antar Variabel RFM")
        st.pyplot(fig6)
        plt.clf()

        # Regresi plot (Actual vs Pred)
        if len(reg_info["y_test"]) > 0:
            fig7, ax = plt.subplots(figsize=(7,5))
            ax.scatter(reg_info["y_test"], reg_info["y_pred_test"], color="steelblue", alpha=0.7, label="Prediksi vs Aktual")
            ax.plot([reg_info["y_test"].min(), reg_info["y_test"].max()], [reg_info["y_test"].min(), reg_info["y_test"].max()], "r--", label="Ideal")
            ax.set_title(f"Regresi Linier Validasi — R²(Test)={reg_info['r2_test']:.2f}")
            ax.set_xlabel("Nilai Aktual (Monetary)")
            ax.set_ylabel("Nilai Prediksi")
            ax.legend()
            st.pyplot(fig7)
            plt.clf()

        # Top 10 Loyalist
        fig8, ax = plt.subplots(figsize=(8,5))
        royal = rfm_k[rfm_k["Segment"]=="Loyalist / High Value"].nlargest(10, "Monetary")
        if not royal.empty:
            ax.barh(royal["Nama Pembeli"], royal["Monetary"], color="gold", edgecolor="black")
            ax.invert_yaxis()
            ax.set_title("Top 10 Pelanggan Loyalist / High Value")
            ax.set_xlabel("Total Pembelian (Monetary)")
            st.pyplot(fig8)
        else:
            st.info("Tidak ada pelanggan kategori 'Loyalist / High Value' untuk ditampilkan.")
        plt.clf()

        # Pie chart segment proportion
        fig9, ax = plt.subplots(figsize=(6,6))
        rfm_k["Segment"].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette("Set3"), startangle=90, ax=ax)
        ax.set_title("Proporsi Pelanggan Berdasarkan Segmen RFM")
        ax.set_ylabel("")
        st.pyplot(fig9)
        plt.clf()

        # Summary text / info
        st.subheader("Ringkasan & Informasi Klaster")
        st.write(f"Jumlah klaster terbentuk: **{rfm_k['Cluster'].nunique()}** (optimal berdasarkan Silhouette Score).")
        st.table(cluster_counts.reset_index().rename(columns={"index":"Cluster", "Cluster":"Count"}).set_index("Cluster"))

        # Save Excel (download)
        st.subheader("Simpan & Unduh Hasil")
        dfs_to_save = {
            "Hasil_RFM_KMeans_Regresi.xlsx": rfm_k,
            "Centroid_KMeans.xlsx": centroids_df
        }
        excel_bytes = to_excel_bytes(dfs_to_save)
        st.download_button(label="Unduh Semua Hasil (Excel)", data=excel_bytes, file_name="Hasil_RFM_KMeans_Regresi_and_Centroid.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.success("Pipeline selesai. Silakan unduh hasil atau lanjutkan analisis.")
    except Exception as e:
        st.exception(f"Terjadi error saat menjalankan pipeline: {e}")

else:
    st.info("Tekan tombol 'Jalankan Analisis (Pipeline Lengkap)' di sidebar untuk memulai.")

# Footer / info
st.markdown("---")
st.markdown("Catatan: Aplikasi ini mengikuti seluruh langkah pada skrip asli Anda: preprocessing, RFM, penentuan cluster (Silhouette), K-Means, titik centroid (skala asli), regresi (Polynomial degree=2 dengan noise kecil pada train), visualisasi, dan penyimpanan hasil.")
