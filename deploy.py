# ============================================================
# 6️⃣ TAB VISUALISASI — VERSI RINGAN
# ============================================================
with tab6:
    st.header("Visualisasi (Versi Ringan – Cepat di Streamlit Cloud)")

    if "rfm_kmeans" not in st.session_state:
        st.warning("Jalankan semua step terlebih dahulu.")
        st.stop()

    rfm = st.session_state["rfm_kmeans"]
    centroids_df = st.session_state["centroids"]

    st.subheader("📌 1. Scatter Plot KMeans (Frequency vs Monetary)")
    fig1, ax1 = plt.subplots(figsize=(7,5))
    sns.scatterplot(
        data=rfm, 
        x="Frequency", 
        y="Monetary", 
        hue="Cluster", 
        palette="tab10",
        s=40
    )
    plt.title("KMeans Clustering (Frequency vs Monetary)")
    st.pyplot(fig1)

    st.subheader("📌 2. Scatter Plot + Centroid")
    fig2, ax2 = plt.subplots(figsize=(7,5))
    plt.scatter(
        rfm["Frequency"], 
        rfm["Monetary"], 
        c=rfm["Cluster"], 
        cmap="tab10", 
        s=40, 
        alpha=0.7
    )
    plt.scatter(
        centroids_df["Frequency"], 
        centroids_df["Monetary"], 
        c="red", 
        s=200, 
        marker="X"
    )
    plt.title("Centroid KMeans")
    plt.xlabel("Frequency")
    plt.ylabel("Monetary")
    st.pyplot(fig2)

    # Plot Regresi
    if "reg_pred" in st.session_state:
        y_test, pred = st.session_state["reg_pred"]

        st.subheader("📌 3. Regresi Linear — Prediksi vs Aktual")
        fig3, ax3 = plt.subplots(figsize=(7,5))
        ax3.scatter(y_test, pred, alpha=0.7)
        ax3.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--"
        )
        ax3.set_title("Regresi: Aktual vs Prediksi")
        ax3.set_xlabel("Nilai Aktual")
        ax3.set_ylabel("Nilai Prediksi")
        st.pyplot(fig3)

    st.success("Visualisasi selesai (versi sangat ringan).")
