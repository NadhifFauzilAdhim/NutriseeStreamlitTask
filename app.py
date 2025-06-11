import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import os
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests

st.set_page_config(
    page_title="NutriSee | Analisis Gizi AI",
    page_icon="🥑",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation_url = "https://lottie.host/80b3b288-3151-4899-849c-609d7842c677/T5B3ymm425.json"
lottie_anim = load_lottieurl(lottie_animation_url)

with st.sidebar:
    st.title("🥑 NutriSee AI")
    if lottie_anim:
        st_lottie(lottie_anim, height=200)
    st.info(
        "Aplikasi ini menggunakan model YOLOv8 untuk mendeteksi makanan "
        "dan menyajikan analisis gizinya secara visual."
    )
    st.markdown("---")
    st.header("Panduan Cepat")
    st.markdown(
        """
        1. **Unggah Gambar**: Seret atau pilih file gambar makanan.
        2. **Mulai Analisis**: Tekan tombol 'Analisis Gambar Makanan'.
        3. **Jelajahi Hasil**: Lihat deteksi dan grafik gizi interaktif.
        """
    )
    st.markdown("---")
    st.caption("© 2025 | Dibuat dengan Streamlit")

# --- JUDUL UTAMA ---
st.markdown("<h1 style='text-align: center;'>Analisis Gizi Makanan Berbasis AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Ambil gambar, unggah, dan biarkan AI kami mengungkap kandungan gizi makanan Anda.</p>", unsafe_allow_html=True)
st.divider()

MODEL_PATH = 'model/best.pt'
NUTRITION_DATA_PATH = 'labels/nutrition_data.csv'

@st.cache_resource
def load_yolo_model(path):
    if not os.path.exists(path):
        st.error(f"File model tidak ditemukan di path: {path}. Pastikan path sudah benar.")
        st.stop()
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        st.stop()

@st.cache_data
def load_nutrition_data(path):
    if not os.path.exists(path):
        st.error(f"File data nutrisi tidak ditemukan di path: {path}. Pastikan path sudah benar.")
        st.stop()
    try:
        df = pd.read_csv(path)
        original_columns = df.columns.tolist()
        df.columns = [col.strip().lower() for col in df.columns]
        if 'class' not in df.columns:
            st.error(f"Error: Kolom 'class' tidak ditemukan di file CSV. Kolom yang ada: {original_columns}")
            st.stop()
        df['class'] = df['class'].str.strip().str.lower()
        return df
    except Exception as e:
        st.error(f"Gagal memuat atau memproses data nutrisi: {e}")
        st.stop()
        
def create_nutrition_chart(protein, carbs, fat):
    """Membuat grafik donat interaktif untuk makronutrien."""
    if any(v == 'N/A' for v in [protein, carbs, fat]):
        return None
        
    labels = ['Protein', 'Karbohidrat', 'Lemak']
    values = [float(protein), float(carbs), float(fat)]
    colors = ['#3EC70B', '#FFC300', '#F94C66'] 

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5,
                                 marker_colors=colors, textinfo='percent',
                                 hoverinfo='label+value+percent')])
    fig.update_layout(
        showlegend=True,
        legend_title_text='Makronutrien',
        font=dict(color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=20, r=20),
        height=300
    )
    return fig

def display_nutrition_info(detected_names, nutrition_df):
    st.markdown("## 📊 **Hasil Analisis Gizi**")
    
    unique_names = sorted(list(set(detected_names))) 
    
    for name in unique_names:
        food_data = nutrition_df[nutrition_df['class'] == name.lower()]

        if not food_data.empty:
            with st.container(border=True):
                food_data_row = food_data.iloc[0]
                st.markdown(f"### **{name.title()}**")
                
                calories = food_data_row.get('calories (kcal)', 'N/A')
                carbs = food_data_row.get('carbohydrates (g)', 'N/A')
                protein = food_data_row.get('protein (g)', 'N/A')
                fat = food_data_row.get('fat (g)', 'N/A')
                fiber = food_data_row.get('fiber (g)', 'N/A')

                col_chart, col_metrics = st.columns([1, 1], gap="large")

                with col_chart:
                    chart = create_nutrition_chart(protein, carbs, fat)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.info("Data makronutrien tidak lengkap untuk membuat grafik.")
                
                with col_metrics:
                    st.metric("⚡ **Kalori**", f"{calories} kcal" if calories != 'N/A' else 'N/A')
                    st.metric("🌾 **Serat**", f"{fiber} g" if fiber != 'N/A' else 'N/A')
                    st.markdown(f"**Protein:** {protein} g")
                    st.markdown(f"**Karbohidrat:** {carbs} g")
                    st.markdown(f"**Lemak:** {fat} g")
        else:
            st.warning(f"⚠️ Data gizi untuk '{name.title()}' tidak ditemukan.")

model = load_yolo_model(MODEL_PATH)
nutrition_df = load_nutrition_data(NUTRITION_DATA_PATH)

col_uploader, col_display = st.columns(2, gap="large")

with col_uploader:
    st.subheader("Langkah 1: Unggah Gambar Anda")
    uploaded_file = st.file_uploader(
        "Pilih file gambar...", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar siap dianalisis", use_container_width=True)
        
        if st.button("🚀 Analisis Gambar Makanan", use_container_width=True, type="primary"):
            with col_display:
                st.subheader("Langkah 2: Lihat Hasilnya")
                with st.spinner("AI sedang menganalisis gambar... 🤖"):
                    image = Image.open(uploaded_file)
                    results = model(image)
                    
                    annotated_image = results[0].plot()
                    annotated_image_rgb = Image.fromarray(annotated_image[..., ::-1])
                    st.image(annotated_image_rgb, caption="Gambar Hasil Deteksi", use_container_width=True)
                    
                    detected_names = [model.names[int(c)] for r in results for c in r.boxes.cls]
                    
                    if detected_names:
                        display_nutrition_info(detected_names, nutrition_df)
                    else:
                        st.success("Gambar berhasil dianalisis, namun tidak ada makanan yang dikenali.")
        
with col_display:
    if not uploaded_file:
        st.info("🖼️ Hasil analisis Anda akan muncul di sini.")