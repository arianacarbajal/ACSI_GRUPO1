import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import tempfile
from scipy.ndimage import zoom
import requests
import os

# Configuración de la página
st.set_page_config(page_title="MRI Visualization and Segmentation", layout="wide")

# Función para descargar el modelo desde Google Drive (con manejo de archivos grandes)
@st.cache_data
def download_model_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    try:
        response = session.get(URL, params={'id': id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, destination)
        st.success(f"Modelo descargado correctamente y guardado en: {destination}")
    except Exception as e:
        st.error(f"Error al descargar el modelo desde Google Drive: {str(e)}")
    return destination

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Descargar el archivo .pth desde Google Drive
model_id = '1r5EWxoBiCMF7ug6jly-3Oma4C9N4ZhGi'  # ID de tu archivo en Drive
model_path = 'modelo_entrenado.pth'
download_model_from_google_drive(model_id, model_path)

# Verificación de la validez del modelo descargado
def is_valid_model_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            first_bytes = f.read(4)
            if first_bytes.startswith(b'\x80\x04'):  # Bytes mágicos para archivos pickle
                return True
            else:
                return False
    except Exception as e:
        st.error(f"Error al verificar el archivo del modelo: {str(e)}")
        return False

# Validar si el modelo es un archivo válido de PyTorch
if model_path and is_valid_model_file(model_path):
    st.write("Archivo del modelo verificado correctamente.")
else:
    st.error("El archivo del modelo no es válido o no se ha descargado correctamente. Revisa el enlace de Google Drive o el archivo .pth.")

# Crear las páginas en la barra lateral
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a", ["Visualización MRI", "Resultados de Segmentación", "Leyendas", "Manual de Usuario", "Planificación Quirúrgica"])

# Función para cargar archivos NIfTI
def load_nifti(file):
    if file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
                temp_file.write(file.read())
                temp_file.flush()
                img = nib.load(temp_file.name)
                return img.get_fdata()
        except Exception as e:
            st.error(f"Error al cargar el archivo NIfTI: {str(e)}")
    return None

# Función de preprocesamiento para manejar volúmenes 4D
def preprocess_volume(volume, target_shape=(128, 128, 128)):
    if len(volume.shape) == 4:
        modalities = volume.shape[-1]
        resized_volumes = []
        
        for i in range(modalities):
            modality_volume = volume[..., i]
            non_zero_coords = np.array(np.nonzero(modality_volume))
            
            if non_zero_coords.size == 0:
                st.error(f"La modalidad {i} no tiene suficientes datos no cero.")
                return None
            
            min_coords = np.min(non_zero_coords, axis=1)
            max_coords = np.max(non_zero_coords, axis=1)
            cropped_volume = modality_volume[min_coords[0]:max_coords[0]+1,
                                             min_coords[1]:max_coords[1]+1,
                                             min_coords[2]:max_coords[2]+1]

            if 0 in cropped_volume.shape:
                st.error(f"Las dimensiones del volumen en la modalidad {i} no son válidas para redimensionar.")
                return None

            st.write(f"Dimensiones del volumen recortado para la modalidad {i}: {cropped_volume.shape}")

            factors = [target / float(dim) for target, dim in zip(target_shape, cropped_volume.shape)]
            try:
                resized_volume = zoom(cropped_volume, factors, order=1)
            except Exception as e:
                st.error(f"Error al redimensionar el volumen en la modalidad {i}: {e}")
                return None
            
            resized_volumes.append(resized_volume)
        
        resized_volume_4d = np.stack(resized_volumes, axis=-1)

        for i in range(modalities):
            non_zero_mask = resized_volume_4d[..., i] > 0
            mean = np.mean(resized_volume_4d[..., i][non_zero_mask])
            std = np.std(resized_volume_4d[..., i][non_zero_mask])
            resized_volume_4d[..., i][non_zero_mask] = (resized_volume_4d[..., i][non_zero_mask] - mean) / std
        
        return resized_volume_4d
    else:
        st.error("El volumen no tiene el número esperado de dimensiones (4D).")
        return None

# Definir el modelo
class SimpleSegmentationModel(torch.nn.Module):
    def __init__(self):
        super(SimpleSegmentationModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Función para mostrar cortes de las imágenes
def plot_mri_slices(data, modality):
    st.subheader(f"{modality} MRI")
    slice_idx = st.slider(f"Selecciona un corte axial para {modality}", 0, data.shape[2] - 1, data.shape[2] // 2)
    fig, ax = plt.subplots()
    ax.imshow(data[:, :, slice_idx], cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

# Cargar el modelo
@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.error(f"El archivo del modelo '{model_path}' no existe.")
        return None

    try:
        model = SimpleSegmentationModel()
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo desde el archivo '{model_path}': {str(e)}")
    
    return None

# Página de Resultados de Segmentación
elif pagina == "Resultados de Segmentación":
    st.title("Resultados de Segmentación")
    st.write("Aquí se mostrarán los resultados de la segmentación del tumor. Sube el archivo de segmentación.")

    img_file = st.file_uploader("Sube el archivo MRI apilado (T1, T2, FLAIR, etc.) en formato NIfTI", type=["nii", "nii.gz"])

    if img_file:
        img_data = load_nifti(img_file)
        if img_data is not None:
            st.write("Imagen cargada correctamente.")
            
            img_preprocessed = preprocess_volume(img_data)

            if img_preprocessed is not None:
                img_tensor = torch.tensor(img_preprocessed).unsqueeze(0).float()

                model = load_model()
                if model:
                    with torch.no_grad():
                        try:
                            pred = model(img_tensor)
                            pred = torch.sigmoid(pred).squeeze().numpy()
                            slice_idx = st.slider("Selecciona un corte axial para visualizar la segmentación", 0, pred.shape[2] - 1, pred.shape[2] // 2)
                            fig, ax = plt.subplots()
                            ax.imshow(pred[:, :, slice_idx], cmap='hot', alpha=0.6)
                            ax.axis('off')
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error al realizar la predicción del modelo: {str(e)}")
                else:
                    st.error("No se pudo cargar el modelo para la segmentación.")

