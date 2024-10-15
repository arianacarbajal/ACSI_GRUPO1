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

# --- Configuración de la página ---
st.set_page_config(page_title="MRI Visualization and Segmentation", layout="wide")

# ---  Configuración del modelo ---
MODEL_ID = '1r5EWxoBiCMF7ug6jly-3Oma4C9N4ZhGi'
MODEL_PATH = 'modelo_entrenado.pth'

# --- Rango de cortes para entrenamiento y segmentación ---
START_SLICE = 40
END_SLICE = 130

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
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ajustar dimensiones en 2D (padding)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
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

def preprocess_volume(volume, target_shape=(128, 128)):
    if len(volume.shape) == 4:  # Verificar que es un volumen 4D (varias modalidades)
        volume = volume[:, :, START_SLICE:END_SLICE, :]  # Recortar el volumen
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
            cropped_volume = modality_volume[
                min_coords[0]: max_coords[0] + 1,
                min_coords[1]: max_coords[1] + 1,
                min_coords[2]: max_coords[2] + 1,
            ]

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

        resized_volume_4d = np.stack(resized_volumes, axis=-1)  # Combinar las modalidades redimensionadas

        st.write(f"Shape del volumen después de preprocess_volume: {resized_volume_4d.shape}")

        for i in range(modalities):
            non_zero_mask = resized_volume_4d[..., i] > 0
            mean = np.mean(resized_volume_4d[..., i][non_zero_mask])
            std = np.std(resized_volume_4d[..., i][non_zero_mask])
            resized_volume_4d[..., i][non_zero_mask] = (resized_volume_4d[..., i][non_zero_mask] - mean) / std

        return resized_volume_4d
    else:
        st.error("El volumen no tiene el número esperado de dimensiones (4D).")
        return None

def plot_mri_slices(data, modality, overlay=None):
    """Muestra cortes axiales de un volumen 3D."""
    st.subheader(f"{modality} MRI")
    slice_idx = st.slider(
        f"Selecciona un corte axial para {modality}",
        0,
        data.shape[2] - 1,
        data.shape[2] // 2,
    )

    fig, ax = plt.subplots()
    ax.imshow(data[:, :, slice_idx], cmap="gray")

    if overlay is not None:
        # Asegúrate de que 'overlay' tiene la misma forma que 'data' en las dos primeras dimensiones
        assert overlay.shape[:2] == data.shape[:2], f"Error: Las formas de la imagen y la máscara no coinciden: {data.shape} vs {overlay.shape}"
        ax.imshow(overlay[:, :, slice_idx], cmap="hot", alpha=0.6)

    ax.axis("off")
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


# --- Lógica principal de la aplicación ---
if __name__ == "__main__":
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

    # --- Página de Visualización MRI ---
    if pagina == "Visualización MRI":
        st.title("Visualización de MRI")
        st.write("Sube los archivos NIfTI de diferentes modalidades para visualizar los cortes.")

        # Subida de las 4 imágenes por separado
        t1_file = st.file_uploader("Sube el archivo T1-weighted (T1)", type=["nii", "nii.gz"])
        t1c_file = st.file_uploader("Sube el archivo T1 con contraste (T1c)", type=["nii", "nii.gz"])
        t2_file = st.file_uploader("Sube el archivo T2-weighted (T2)", type=["nii", "nii.gz"])
        flair_file = st.file_uploader("Sube el archivo T2-FLAIR", type=["nii", "nii.gz"])

        # Visualizar cada imagen por separado si se ha subido
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

    # --- Página de Resultados de Segmentación ---
    elif pagina == "Resultados de Segmentación":
        st.title("Resultados de Segmentación")
        st.write(
            "Aquí se mostrarán los resultados de la segmentación del tumor. Sube el archivo MRI apilado para segmentar."
        )

        # Subida del archivo apilado 
        uploaded_file = st.file_uploader(
            "Sube el archivo MRI apilado (T1, T2, T1c, FLAIR) en formato NIfTI",
            type=["nii", "nii.gz"],
        )

        if uploaded_file is not None:
            img_data = load_nifti(uploaded_file)

            if img_data is not None:
                st.write("Imagen cargada correctamente.")

                try:
                    # --- Preprocesamiento ---
                    img_preprocessed = preprocess_volume(img_data)
                    st.write(f"Shape después de preprocess_volume: {img_preprocessed.shape}")

                    if img_preprocessed is not None and model is not None:
                        st.write("Realizando la segmentación...")

                        with torch.no_grad():
                            # --- Convertir a tensor de PyTorch y agregar dimensión de batch ---
                            img_tensor = torch.tensor(img_preprocessed).unsqueeze(0).float()  # [1, 128, 128, 90, 4]

                            # --- Seleccionar un corte para la segmentación 2D ---
                            slice_idx = st.slider(
                                "Selecciona un corte axial para la segmentación",
                                0,
                                img_tensor.shape[3] - 1,
                                img_tensor.shape[3] // 2,
                            )
                            # Extraer el slice de cada modalidad en el corte seleccionado
                            img_slice = img_tensor[:, :, :, slice_idx, :] # [1, 128, 128, 4]
                            img_slice = img_slice.permute(0, 3, 1, 2)  # Reordenar para que las modalidades sean canales: [1, 4, 128, 128]
                            # --- Realizar la predicción ---
                            pred = model(img_slice)
                            pred = torch.sigmoid(pred).squeeze().cpu().numpy() 

                        # --- Visualización (T1 original + segmentación) ---
                        plot_mri_slices(img_preprocessed[:, :, slice_idx, 0], "T1 Original", overlay=pred)

                except Exception as e:
                    st.error(f"Error durante la segmentación: {e}")
                    st.write(traceback.format_exc())

            else:
                st.error("Error al cargar la imagen.")

    # --- (Resto del código para "Leyendas", "Manual de Usuario", etc.)
 
    # --- Página de Leyendas ---
    elif pagina == "Leyendas":
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

    # --- Página del Manual de Usuario ---
    elif pagina == "Manual de Usuario":
        st.title("Manual de Usuario")
        st.write(
            """
        Manual de Uso del Visualizador de MRI:

        1. Cargar Archivos: 
            - Para visualizar:  Sube los archivos MRI en formato NIfTI para cada modalidad (T1, T2, T1c, FLAIR) en la página "Visualización MRI". Puedes subir un único archivo que contenga todas las modalidades o cada modalidad por separado. 
            - Para segmentar: Sube un único archivo que contenga las 4 modalidades (T1, T2, T1c, FLAIR) en la página "Resultados de Segmentación".
        2. Visualización de Cortes: Usa el control deslizante para seleccionar el corte axial que desees visualizar.
        3. Segmentación: Una vez que hayas cargado un archivo válido, la segmentación se ejecutará automáticamente y se mostrará junto a la imagen original.
        4. Interpretación: Utiliza la página de Leyendas para entender el significado de los colores en la segmentación.
        5. Planificación Quirúrgica: La página "Planificación Quirúrgica" proporciona información sobre cómo la segmentación puede ayudar en la planificación de cirugías.
        """
        )

    # --- Página sobre Planificación Quirúrgica ---
    elif pagina == "Planificación Quirúrgica":
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

    # --- Mensaje de pie de página ---
    st.sidebar.markdown("---")
    st.sidebar.info("Desarrollado por el Grupo 1 de ACSI")
