import streamlit as st
import pandas as pd
import datetime
import joblib
import plotly.express as px

st.set_page_config(page_title="Prediksi & Analisis Harga Rumah", layout="wide")

# Load model & data
model = joblib.load("random_forest_model.pkl")
df = pd.read_excel("rumah_fix.xlsx")

# Layout
col_pred, spacer, col_viz = st.columns([1.2, 0.2, 2])

# ======================== KIRI: FORM PREDIKSI ========================
with col_pred:
    st.title("Prediksi Harga Rumah")

    # Baris 1
    col1, col2 = st.columns(2)
    with col1:
        bedrooms = st.number_input("Jumlah Kamar Tidur", min_value=0, value=3)
    with col2:
        bathrooms = st.number_input("Jumlah Kamar Mandi", min_value=0, value=2)

    # Baris 2
    col3, col4 = st.columns(2)
    with col3:
        land_size = st.number_input("Luas Tanah (m²)", min_value=0, value=100)
    with col4:
        building_size = st.number_input("Luas Bangunan (m²)", min_value=0, value=80)

    # Baris 3
    col5, col6 = st.columns(2)
    with col5:
        carports = st.number_input("Jumlah Carports", min_value=0, value=1)
    with col6:
        floors = st.number_input("Jumlah Lantai", min_value=1, value=1)

    # Baris 4
    col7, col8 = st.columns(2)
    with col7:
        electricity = st.selectbox("Listrik (VA)", [1300, 2200, 3500, 4400, 5500])
    with col8:
        furnishing = st.selectbox("Tingkat Furnishing", ["Furnished", "Semi Furnished", "Unfurnished"])

    # Baris 5
    year_built = st.number_input("Tahun Dibangun", min_value=1950, max_value=datetime.datetime.now().year, value=2020)

    # Prediksi
    furnishing_map = {"Furnished": 2, "Semi Furnished": 1, "Unfurnished": 0}
    furnishing_encoded = furnishing_map[furnishing]
    building_age = datetime.datetime.now().year - year_built

    input_data = pd.DataFrame([[bedrooms, bathrooms, land_size, building_size,
                                carports, floors, electricity, furnishing_encoded,
                                year_built, building_age, -6.2, 106.8, 0, 0, 1]],
                              columns=['bedrooms', 'bathrooms', 'land_size_m2', 'building_size_m2',
                                       'carports', 'floors', 'electricity', 'furnishing',
                                       'year_built', 'building_age', 'lat', 'long',
                                       'maid_bedrooms', 'maid_bathrooms', 'garages'])

    btn_col, result_col = st.columns([1, 2])
    with btn_col:
        if st.button("🔮 Prediksi"):
            try:
                prediction = model.predict(input_data)[0]
                with result_col:
                    st.success(f"💰 Estimasi Harga: **Rp {int(prediction):,}**")
            except Exception as e:
                st.error(f"Gagal prediksi: {e}")

    st.markdown("<br><br>", unsafe_allow_html=True)

# ======================== KANAN: VISUALISASI ========================
with col_viz:
    st.title("Analisis Data")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📍 Rata-rata per Kota",
        "📈 Tren Tahun Dibangun",
        "🏘️ Jumlah Listing",
        "📊 Korelasi Fitur"
    ])

    # Tab 1
    with tab1:
        if 'city' in df.columns:
            top_kota = st.slider("Tampilkan Top Kota Berdasarkan Harga Rata-rata", 0, 10, 5)
            avg_price_city = df.groupby("city")["price_in_rp"].mean().sort_values(ascending=False).head(top_kota).reset_index()
            fig1 = px.bar(avg_price_city[::-1], x='price_in_rp', y='city',
                          orientation='h',
                          labels={'price_in_rp': 'Harga Rata-rata (Rp)', 'city': 'Kota'},
                          title=f"Top {top_kota} Kota dengan Harga Rumah Rata-rata Tertinggi")
            fig1.update_layout(yaxis=dict(dtick=1), title_x=0.3)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("Kolom 'city' tidak tersedia di dataset.")

    # Tab 2
    with tab2:
        if 'year_built' in df.columns and 'price_in_rp' in df.columns and 'city' in df.columns:
            df_valid = df.dropna(subset=['year_built', 'price_in_rp', 'city'])
            trend_data = df_valid.groupby(['year_built', 'city'])['price_in_rp'].mean().reset_index()
            fig2 = px.line(trend_data, x='year_built', y='price_in_rp', color='city',
                           markers=True,
                           labels={'year_built': 'Tahun Dibangun', 'price_in_rp': 'Harga Rata-rata (Rp)', 'city': 'Kota'},
                           title="📈 Tren Harga Rata-rata Rumah Berdasarkan Tahun Dibangun dan Kota")
            fig2.update_layout(title_x=0.3)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Dataset tidak memiliki kolom 'year_built', 'price_in_rp', atau 'city'.")

    # Tab 3
    with tab3:
        if 'city' in df.columns:
            count_city = df['city'].value_counts().reset_index()
            count_city.columns = ['Kota', 'Jumlah']
            fig3 = px.bar(count_city, x='Kota', y='Jumlah',
                          title="Jumlah Listing Rumah per Kota",
                          labels={'Jumlah': 'Jumlah Rumah'})
            fig3.update_layout(xaxis_tickangle=45, title_x=0.3)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Kolom 'city' tidak tersedia di dataset.")

    # Tab 4 - Korelasi
    with tab4:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if len(numerical_cols) >= 2:
            st.subheader("📊 Matriks Korelasi Antar Fitur Numerik")
            selected_cols = st.multiselect("Pilih Fitur untuk Korelasi", numerical_cols, default=numerical_cols)

            if len(selected_cols) >= 2:
                corr_matrix = df[selected_cols].corr()
                fig4 = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    origin='lower',
                    labels=dict(color="Korelasi"),
                    title="Heatmap Korelasi"
                )
                fig4.update_layout(
                    title_x=0.3,
                    width=900,  # Lebar heatmap
                    height=800, # Tinggi heatmap
                    margin=dict(l=100, r=100, t=100, b=100)
                )
                st.plotly_chart(fig4, use_container_width=False)
            else:
                st.info("Pilih minimal 2 fitur untuk ditampilkan.")
        else:
            st.warning("Tidak ditemukan cukup fitur numerik untuk membuat korelasi.")
