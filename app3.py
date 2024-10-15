import streamlit as st 
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import tempfile
from scipy.ndimage import zoom
import requests  # Para descargar el modelo desde Google Drive

# Función para descargar el modelo desde Google Drive
def download_model(url, output_path):
    response = requests.get(url)
    with open(output_path, 'wb') as f:
        f.write(response.content)

# Descargar el archivo .pth desde Google Drive
model_url = 'https://drive.google.com/uc?export=download&id=1r5EWxoBiCMF7ug6jly-3Oma4C9N4ZhGi'
download_model(model_url, 'modelo_entrenado.pth')

# Crear las páginas en la barra lateral
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a", ["Visualización MRI", "Resultados de Segmentación", "Leyendas", "Manual de Usuario", "Planificación Quirúrgica"])

# Función para cargar archivos NIfTI
def load_nifti(file):
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
            temp_file.write(file.read())  # Escribir el archivo en el sistema temporal
            temp_file.flush()  # Asegurarse de que el archivo esté completamente escrito
            img = nib.load(temp_file.name)  # Cargar el archivo NIfTI desde el archivo temporal
            return img.get_fdata()
    return None

# Función de preprocesamiento para manejar volúmenes 4D
def preprocess_volume(volume, target_shape=(128, 128, 128)):
    """Recorta y normaliza el volumen, ajustándolo al tamaño objetivo. Maneja volúmenes 4D (modalidades)."""
    
    if len(volume.shape) == 4:  # Verificar si el volumen es 4D (x, y, z, canales)
        modalities = volume.shape[-1]  # Número de modalidades
        resized_volumes = []
        
        for i in range(modalities):  # Procesar cada modalidad individualmente
            modality_volume = volume[..., i]  # Extraer una modalidad
            non_zero_coords = np.array(np.nonzero(modality_volume))
            
            if non_zero_coords.size == 0:
                st.error(f"La modalidad {i} no tiene suficientes datos no cero.")
                return None
            
            min_coords = np.min(non_zero_coords, axis=1)
            max_coords = np.max(non_zero_coords, axis=1)
            cropped_volume = modality_volume[min_coords[0]:max_coords[0]+1,
                                             min_coords[1]:max_coords[1]+1,
                                             min_coords[2]:max_coords[2]+1]

            if cropped_volume.shape[0] == 0 or cropped_volume.shape[1] == 0 or cropped_volume.shape[2] == 0:
                st.error(f"Las dimensiones del volumen en la modalidad {i} no son válidas para redimensionar.")
                return None

            st.write(f"Dimensiones del volumen recortado para la modalidad {i}: {cropped_volume.shape}")

            # Redimensionar al tamaño objetivo
            factors = [target / float(dim) for target, dim in zip(target_shape, cropped_volume.shape)]
            try:
                resized_volume = zoom(cropped_volume, factors, order=1)
            except Exception as e:
                st.error(f"Error al redimensionar el volumen en la modalidad {i}: {e}")
                return None
            
            resized_volumes.append(resized_volume)
        
        # Combinar las modalidades redimensionadas en un solo volumen 4D
        resized_volume_4d = np.stack(resized_volumes, axis=-1)

        # Normalizar cada modalidad por separado
        for i in range(modalities):
            non_zero_mask = resized_volume_4d[..., i] > 0
            mean = np.mean(resized_volume_4d[..., i][non_zero_mask])
            std = np.std(resized_volume_4d[..., i][non_zero_mask])
            resized_volume_4d[..., i][non_zero_mask] = (resized_volume_4d[..., i][non_zero_mask] - mean) / std
        
        return resized_volume_4d
    
    else:
        st.error("El volumen no tiene el número esperado de dimensiones (4D).")
        return None

# Definir el modelo (Ejemplo de UNet simple)
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
    plt.imshow(data[:, :, slice_idx], cmap='gray')
    plt.axis('off')
    st.pyplot(plt)

# Cargar el modelo
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = "modelo_entrenado.pth"
    
    if not os.path.exists(model_path):
        st.error(f"El archivo del modelo '{model_path}' no existe.")
        return None

    try:
        model = SimpleSegmentationModel()
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except RuntimeError as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.info("Este error puede ocurrir si el modelo fue guardado con una versión diferente de PyTorch.")
    except _pickle.UnpicklingError as e:
        st.error(f"Error al deserializar el modelo: {str(e)}")
        st.info("El archivo del modelo podría estar corrupto o ser incompatible.")
    except Exception as e:
        st.error(f"Error inesperado al cargar el modelo: {str(e)}")
    
    return None

# Uso de la función
model = load_model()
if model is None:
    st.warning("No se pudo cargar el modelo. Algunas funcionalidades pueden no estar disponibles.")
else:
    st.success("Modelo cargado exitosamente.")

# Página de visualización MRI
if pagina == "Visualización MRI":
    st.title("Visualización de MRI")
    st.write("Sube los archivos NIfTI de diferentes modalidades para visualizar los cortes.")

    t1_file = st.file_uploader("Sube el archivo T1-weighted (T1)", type=["nii", "nii.gz"])
    t1c_file = st.file_uploader("Sube el archivo T1 con contraste (T1c)", type=["nii", "nii.gz"])
    t2_file = st.file_uploader("Sube el archivo T2-weighted (T2)", type=["nii", "nii.gz"])
    flair_file = st.file_uploader("Sube el archivo T2-FLAIR", type=["nii", "nii.gz"])

    if t1_file or t1c_file or t2_file or flair_file:
        if t1_file:
            t1_data = load_nifti(t1_file)
            if t1_data is not None:
                plot_mri_slices(t1_data, "T1-weighted")

        if t1c_file:
            t1c_data = load_nifti(t1c_file)
            if t1c_data is not None:
                plot_mri_slices(t1c_data, "T1c (con contraste)")

        if t2_file:
            t2_data = load_nifti(t2_file)
            if t2_data is not None:
                plot_mri_slices(t2_data, "T2-weighted")

        if flair_file:
            flair_data = load_nifti(flair_file)
            if flair_data is not None:
                plot_mri_slices(flair_data, "T2-FLAIR")

# Página de Resultados de Segmentación
elif pagina == "Resultados de Segmentación":
    st.title("Resultados de Segmentación")
    st.write("Aquí se mostrarán los resultados de la segmentación del tumor. Sube el archivo de segmentación.")

    img_file = st.file_uploader("Sube el archivo MRI apilado (T1, T2, FLAIR, etc.) en formato NIfTI", type=["nii", "nii.gz"])

    if img_file:
        img_data = load_nifti(img_file)
        if img_data is not None:
            st.write("Imagen cargada correctamente.")
            
            # Preprocesar la imagen
            img_preprocessed = preprocess_volume(img_data)

            if img_preprocessed is not None:
                # Convertir la imagen preprocesada en un tensor y añadir la dimensión de batch
                img_tensor = torch.tensor(img_preprocessed).unsqueeze(0).float()

                # Cargar el modelo
                model = load_model()

                # Realizar la segmentación
                with torch.no_grad():
                    pred = model(img_tensor)
                    pred = torch.sigmoid(pred).squeeze().numpy()

                # Mostrar la segmentación
                slice_idx = st.slider("Selecciona un corte axial para visualizar la segmentación", 0, pred.shape[2] - 1, pred.shape[2] // 2)
                plt.imshow(pred[:, :, slice_idx], cmap='hot', alpha=0.6)
                plt.axis('off')
                st.pyplot(plt)

# Página de Leyendas
elif pagina == "Leyendas":
    st.title("Leyendas de Segmentación")
    st.write("""
    En las imágenes segmentadas, cada valor representa un tipo de tejido. A continuación se muestra la leyenda para interpretar las imágenes:

    - 0: Fondo
    - 1: Núcleo de tumor necrótico (azul)
    - 2: Tumor realzado (amarillo)
    - 3: Tejido edematoso peritumoral (verde)
    """)

# Página del Manual de Usuario
elif pagina == "Manual de Usuario":
    st.title("Manual de Usuario")
    st.write("""
    Manual de Uso del Visualizador de MRI:

    1. Cargar Archivos: Sube los archivos MRI en formato NIfTI para cada modalidad (T1, T2, T1c, FLAIR).
    2. Visualización de Cortes: Usa el control deslizante para seleccionar el corte axial que desees visualizar.
    3. Segmentación: Sube el archivo de segmentación para visualizar las etiquetas correspondientes al tumor.
    """)

# Página sobre Planificación Quirúrgica
elif pagina == "Planificación Quirúrgica":
    st.title("Aplicaciones en la Planificación Quirúrgica")
    st.write("""
    La segmentación de imágenes cerebrales juega un papel crucial en la planificación de cirugías para la resección de tumores cerebrales. 
    Al identificar el núcleo del tumor, el tejido edematoso y el tumor realzado, los cirujanos pueden planificar estrategias precisas para la intervención quirúrgica.

    Este sistema de visualización y segmentación permite a los médicos observar la estructura del tumor en detalle y tomar decisiones informadas sobre la ruta quirúrgica y el tratamiento más adecuado.
    """)


