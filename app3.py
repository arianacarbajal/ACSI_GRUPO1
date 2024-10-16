import streamlit as st 
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
from scipy.ndimage import zoom
import os
import gdown
import traceback
import io

# --- Configuración de la página ---
st.set_page_config(page_title="MRI Visualization and Segmentation", layout="wide")

# ---  Configuración del modelo ---
MODEL_ID = '1r5EWxoBiCMF7ug6jly-3Oma4C9N4ZhGi'
MODEL_PATH = 'modelo_entrenado.pth'

# --- Definición del modelo U-Net 2D ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=3):
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
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1) 

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

# --- Funciones auxiliares ---
@st.cache_data
def download_model_from_gdrive(model_id, model_path):
    try:
        gdown.download(f'https://drive.google.com/uc?id={model_id}', model_path, quiet=False)
        st.success(f"Modelo descargado y guardado en {model_path}")
    except Exception as e:
        st.error(f"Error al descargar el modelo: {str(e)}")
    return model_path

def load_nifti1(file):
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
            temp_file.write(file.read())
            temp_file.flush()
            img = nib.load(temp_file.name)
            return img.get_fdata()
    return None

def plot_mri_slices1(data, modality):
    st.subheader(f"{modality} MRI")
    slice_idx = st.slider(f"Selecciona un corte axial para {modality}", 0, data.shape[2] - 1, data.shape[2] // 2)
    fig, ax = plt.subplots()
    ax.imshow(data[:, :, slice_idx], cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

@st.cache_resource
def load_model():
    st.write("Cargando el modelo...") 
    if not os.path.exists(MODEL_PATH):
        st.error(f"El archivo del modelo '{MODEL_PATH}' no existe. Descargando...")
        download_model_from_gdrive(MODEL_ID, MODEL_PATH)

    try:
        model = UNet(n_channels=4, n_classes=3) 
        st.write(f"Intentando cargar el modelo desde {MODEL_PATH}...")
        state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        st.success("Modelo cargado correctamente.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.write(traceback.format_exc())
    return None

# --- Funciones para cada página ---
def visualizacion_mri():
    st.title("Visualización de MRI")
    st.write("Sube los archivos NIfTI de diferentes modalidades para visualizar los cortes.")

    t1_file = st.file_uploader("Sube el archivo T1-weighted (T1)", type=["nii", "nii.gz"])
    t1c_file = st.file_uploader("Sube el archivo T1 con contraste (T1c)", type=["nii", "nii.gz"])
    t2_file = st.file_uploader("Sube el archivo T2-weighted (T2)", type=["nii", "nii.gz"])
    flair_file = st.file_uploader("Sube el archivo T2-FLAIR", type=["nii", "nii.gz"])

    if t1_file or t1c_file or t2_file or flair_file:
        if t1_file:
            t1_data = load_nifti1(t1_file)
            if t1_data is not None:
                plot_mri_slices1(t1_data, "T1-weighted")

        if t1c_file:
            t1c_data = load_nifti1(t1c_file)
            if t1c_data is not None:
                plot_mri_slices1(t1c_data, "T1c (con contraste)")

        if t2_file:
            t2_data = load_nifti1(t2_file)
            if t2_data is not None:
                plot_mri_slices1(t2_data, "T2-weighted")

        if flair_file:
            flair_data = load_nifti1(flair_file)
            if flair_data is not None:
                plot_mri_slices1(flair_data, "T2-FLAIR")

def resultados_segmentacion():
    st.title("Resultados de Segmentación")
    st.write("Aquí se mostrarán los resultados de la segmentación del tumor. Sube el archivo apilado (stack) para segmentar.")

    uploaded_stack = st.file_uploader(
        "Sube el archivo apilado de MRI (.npy o .nii/.nii.gz)", 
        type=["npy", "nii", "nii.gz"]
    )

    if uploaded_stack is not None:
        try:
            # Cargar datos 
            if uploaded_stack.name.endswith('.npy'):
                img_data = np.load(uploaded_stack)
            elif uploaded_stack.name.endswith(('.nii', '.nii.gz')):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
                    temp_file.write(uploaded_stack.read())
                    temp_file.flush() 
                    nii_img = nib.load(temp_file.name)
                    img_data = nii_img.get_fdata()  
                    st.write("Archivo NIfTI cargado correctamente.")
                os.remove(temp_file.name)
            else:
                st.error("Tipo de archivo no soportado. Por favor, carga un archivo .npy o .nii/.nii.gz.")
                st.stop()  

            # Comprobaciones de dimensiones
            if len(img_data.shape) != 4:
                raise ValueError(f"Error: Se esperaban 4 dimensiones (alto, ancho, profundidad, canales). Se obtuvieron: {img_data.shape}")

            # Aquí iría el código para preprocesar y segmentar la imagen
            # Por ahora, solo mostraremos un mensaje
            st.write("Imagen cargada correctamente. La segmentación se implementará en futuras versiones.")

        except Exception as e:
            st.error(f"Error durante la carga o segmentación: {e}")
            st.write(traceback.format_exc())

def leyendas():
    st.title("Leyendas de Segmentación")
    st.write(
        """
    En las imágenes segmentadas, cada valor representa un tipo de tejido. A continuación se muestra la leyenda para interpretar las imágenes:

    - 0: Fondo
    - 1: Núcleo de tumor necrótico (rojo)
    - 2: Tumor realzado (amarillo)
    - 3: Tejido edematoso peritumoral (verde)
    """
    )

def manual_usuario():
    st.title("Manual de Usuario")
    st.write(
        """
    Manual de Uso del Visualizador de MRI:

    1. Cargar Archivos: 
        - Para visualizar: Sube los archivos MRI en formato NIfTI para cada modalidad (T1, T2, T1c, FLAIR) en la página "Visualización MRI".
        - Para segmentar: Sube un único archivo que contenga las 4 modalidades (T1, T2, T1c, FLAIR) en la página "Resultados de Segmentación".
    2. Visualización de Cortes: Usa el control deslizante para seleccionar el corte axial que desees visualizar.
    3. Segmentación: Una vez que hayas cargado un archivo válido, la segmentación se ejecutará automáticamente y se mostrará junto a la imagen original.
    4. Interpretación: Utiliza la página de Leyendas para entender el significado de los colores en la segmentación.
    5. Planificación Quirúrgica: La página "Planificación Quirúrgica" proporciona información sobre cómo la segmentación puede ayudar en la planificación de cirugías.
    """
    )

def planificacion_quirurgica():
    st.title("Aplicaciones en la Planificación Quirúrgica")
    st.write(
        """
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

    **Recuerda que esta herramienta es un apoyo a la decisión clínica y debe utilizarse en conjunto con la experiencia del neurocirujano y otros datos clínicos relevantes.**
    """
    )

# --- Lógica principal de la aplicación ---
def main():
    model = load_model()

    # Barra lateral
    st.sidebar.title("Navegación")
    pagina = st.sidebar.radio(
        "Ir a",
        [
            "Visualización MRI",
            "Resultados de Segmentación",
            "Leyendas",
            "Manual de Usuario",
            "Planificación Quirúrgica",
        ],
    )

    # Navegación
    if pagina == "Visualización MRI":
        visualizacion_mri()
    elif pagina == "Resultados de Segmentación":
        resultados_segmentacion()
    elif pagina == "Leyendas":
        leyendas()
    elif pagina == "Manual de Usuario":
        manual_usuario()
    elif pagina == "Planificación Quirúrgica":
        planificacion_quirurgica()

if __name__ == "__main__":
    main()
