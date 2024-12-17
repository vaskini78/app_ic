import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Caminho do modelo treinado
MODEL_PATH = 'C:/Users/rocha/OneDrive/Ambiente de Trabalho/vasco/IC_Project_Fase3_Grid_SaveModel.h5'
IM_SIZE = 32  # Tamanho da imagem esperado pelo modelo
LABELS = ['using_laptop', 'clapping', 'eating', 'drinking']  # Nomes das classes

# Carregar o modelo salvo
@st.cache(allow_output_mutation=True)
def load_trained_model():
    model = load_model(MODEL_PATH)
    return model

model = load_trained_model()

# Função para processar a imagem
def preprocess_image(image_file):
    try:
        img = load_img(image_file, target_size=(IM_SIZE, IM_SIZE))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalização
        return img_array
    except Exception as e:
        st.error(f"Erro ao processar a imagem: {e}")
        return None

# Interface Streamlit
st.title("Classificação de Imagens com Modelo Treinado")
st.write("Envie uma imagem para classificação.")

# Upload da imagem
uploaded_image = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Exibir a imagem carregada
    st.image(uploaded_image, caption="Imagem carregada", use_column_width=True)
    st.write("Classificando...")

    # Processar a imagem
    img_array = preprocess_image(uploaded_image)
    if img_array is not None:
        # Fazer a predição
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions, axis=1)[0]

        # Exibir resultados
        st.success(f"Classe prevista: {LABELS[predicted_class]}")
        st.info(f"Confiança: {confidence:.2f}")
