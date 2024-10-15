import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import tempfile
from scipy.ndimage import zoom
import os
import gdown

# Configuración de la página
st.set_page_config(page_title="MRI Visualization and Segmentation", layout="wide")

# Función para descargar el modelo desde Google Drive
@st.cache_data
def download_model_from_gdrive(model_id, model_path):
    try:
        gdown.download(f'https://drive.google.com/uc?id={model_id}', model_path, quiet=False)
        st.success(f"Modelo descargado y guardado en {model_path}")
    except Exception as e:
        st.error(f"Error al descargar el modelo: {str(e)}")
    return model_path

# Descargar el archivo .pth desde Google Drive
model_id = '1r5EWxoBiCMF7ug6jly-3Oma4C9N4ZhGi'
model_path = 'modelo_entrenado.pth'
st.write("Intentando descargar el modelo...")
model_path = download_model_from_gdrive(model_id, model_path)

# Definir la arquitectura U-Net
class UNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Definir bloques de la U-Net (convoluciones dobles, upsampling, etc.)
class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = torch.nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Cargar el modelo con manejo de discrepancias
@st.cache_resource
def load_model():
    st.write("Cargando el modelo...")
    if not os.path.exists(model_path):
        st.error(f"El archivo del modelo '{model_path}' no existe.")
        return None

    try:
        model = UNet(n_channels=4, n_classes=3)
        st.write(f"Intentando cargar el modelo desde {model_path}...")

        # Cargar el estado del modelo y manejar discrepancias
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # Filtrar los pesos inesperados
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
        model.load_state_dict(filtered_state_dict, strict=False)

        model.eval()
        st.success("Modelo cargado correctamente.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
    
    return None

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

# Función para mostrar cortes de las imágenes
def plot_mri_slices(data, modality):
    st.subheader(f"{modality} MRI")
    slice_idx = st.slider(f"Selecciona un corte axial para {modality}", 0, data.shape[2] - 1, data.shape[2] // 2)
    fig, ax = plt.subplots()
    ax.imshow(data[:, :, slice_idx], cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

# Página de visualización MRI
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a", ["Visualización MRI", "Resultados de Segmentación", "Leyendas", "Manual de Usuario", "Planificación Quirúrgica"])

if pagina == "Visualización MRI":
    st.title("Visualización de MRI")
    st.write("Sube los archivos NIfTI de diferentes modalidades para visualizar los cortes.")

    t1_file = st.file_uploader("Sube el archivo T1-weighted (T1)", type=["nii", "nii.gz"])
    t1c_file = st.file_uploader("Sube el archivo T1 con contraste (T1c)", type=["nii", "nii.gz"])
    t2_file = st.file_uploader("Sube el archivo T2-weighted (T2)", type=["nii", "nii.gz"])
    flair_file = st.file_uploader("Sube el archivo T2-FLAIR", type=["nii", "nii.gz"])

    if t1_file or t1c_file or t2_file or flair_file:
        col1, col2 = st.columns(2)
        with col1:
            if t1_file:
                t1_data = load_nifti(t1_file)
                if t1_data is not None:
                    plot_mri_slices(t1_data, "T1-weighted")
            if t2_file:
                t2_data = load_nifti(t2_file)
                if t2_data is not None:
                    plot_mri_slices(t2_data, "T2-weighted")
        with col2:
            if t1c_file:
                t1c_data = load_nifti(t1c_file)
                if t1c_data is not None:
                    plot_mri_slices(t1c_data, "T1c (con contraste)")
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
            
            img_preprocessed = preprocess_volume(img_data)

            if img_preprocessed is not None:
                img_tensor = torch.tensor(img_preprocessed).unsqueeze(0).float()

                model = load_model()
                if model:
                    with torch.no_grad():
                        pred = model(img_tensor)
                        pred = torch.sigmoid(pred).squeeze().numpy()

                    slice_idx = st.slider("Selecciona un corte axial para visualizar la segmentación", 0, pred.shape[2] - 1, pred.shape[2] // 2)
                    fig, ax = plt.subplots()
                    ax.imshow(pred[:, :, slice_idx], cmap='hot', alpha=0.6)
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.error("No se pudo cargar el modelo para la segmentación.")

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
    4. Interpretación: Utiliza la página de Leyendas para entender el significado de los colores en la segmentación.
    5. Planificación Quirúrgica: Consulta la información sobre cómo estos datos pueden ayudar en la planificación de cirugías.
    """)

# Página sobre Planificación Quirúrgica
elif pagina == "Planificación Quirúrgica":
    st.title("Aplicaciones en la Planificación Quirúrgica")
    st.write("""
    La segmentación de imágenes cerebrales juega un papel crucial en la planificación de cirugías para la resección de tumores cerebrales. 
    Al identificar el núcleo del tumor, el tejido edematoso y el tumor realzado, los cirujanos pueden planificar estrategias precisas para la intervención quirúrgica.

    Este sistema de visualización y segmentación permite a los médicos:
    1. Observar la estructura del tumor en detalle.
    2. Identificar áreas críticas y zonas de riesgo.
    3. Planificar la ruta quirúrgica más segura y efectiva.
    4. Estimar el volumen del tumor y la extensión de la resección necesaria.
    5. Evaluar la proximidad del tumor a estructuras cerebrales importantes.

    La precisión de esta información es vital para:
    - Maximizar la extirpación del tumor.
    - Minimizar el daño al tejido cerebral sano.
    - Mejorar los resultados postoperatorios del paciente.
    - Facilitar la comunicación entre el equipo médico y con el paciente.
    """)

# Mensaje de pie de página
st.sidebar.markdown("---")
st.sidebar.info("Desarrollado por el Grupo 1 de ACSI")


