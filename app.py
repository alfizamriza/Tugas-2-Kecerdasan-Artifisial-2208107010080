import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Memuat model
model = load_model('model_50.h5')

# Fungsi untuk memproses dan memprediksi gambar
def predict_image(image):
    # Ubah ukuran gambar ke dimensi yang sesuai
    image = image.resize((200, 200))  # Sesuaikan dengan input model
    img_array = img_to_array(image) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension
    
    # Prediksi menggunakan model
    prediction = model.predict(img_array)
    class_index = (prediction[0][0] > 0.5).astype(int)  # Biner: 0 atau 1
    confidence = prediction[0][0] if class_index == 1 else 1 - prediction[0][0]
    return class_index, confidence

# Streamlit UI
st.title("Dandelion vs Grass Classifier")
st.write("Unggah gambar untuk mengetahui apakah itu Dandelion atau Rumput.")

# Upload file
uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
    
    # Load gambar menggunakan PIL
    image = Image.open(uploaded_file)
    
    # Prediksi
    st.write("Sedang memproses...")
    
    try:
        class_index, confidence = predict_image(image)
        
        # Tampilkan hasil prediksi
        if class_index == 1:
            st.success(f"Gambar ini kemungkinan besar adalah **Rumput** dengan tingkat kepercayaan {confidence:.2f}.")
        else:
            st.success(f"Gambar ini kemungkinan besar adalah **Dandelion** dengan tingkat kepercayaan {confidence:.2f}.")
    except ValueError as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.write("Pastikan gambar sesuai dengan dimensi input yang diharapkan model Anda.")
