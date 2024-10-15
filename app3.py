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

# --- Definición del modelo U-Net 3D ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels), #<- Cambio importante para 3D
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels), #<- Cambio importante para 3D
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2), #<- Cambio importante para 3D
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True) #<- Adaptado para 3D
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2): 
        x1 = self.up(x1)
        # Ajustar dimensiones si es necesario (padding) -  Asegúrate de que esto funciona correctamente en 3D
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4] #<- Añadida dimension Z
        x1 = F.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2]
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
        self.outc = nn.Conv3d(64, n_classes, kernel_size=1) #<- Cambio importante para 3D

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
            cropped_volume = modality_volume[
                            min_coords[0] : max_coords[0] + 1,
                            min_coords[1] : max_coords[1] + 1,
                            min_coords[2] : max_coords[2] + 1,
                        ]

            if 0 in cropped_volume.shape:
                st.error(
                    f"Las dimensiones del volumen en la modalidad {i} no son válidas para redimensionar."
                )
                return None

            st.write(
                f"Dimensiones del volumen recortado para la modalidad {i}: {cropped_volume.shape}"
            )

            factors = [target / float(dim) for target, dim in zip(target_shape, cropped_volume.shape)]
            try:
                resized_volume = zoom(cropped_volume, factors, order=1)
            except Exception as e:
                st.error(f"Error al redimensionar el volumen en la modalidad {i}: {e}")
                return None

            resized_volumes.append(resized_volume)

        resized_volume_4d = np.stack(resized_volumes, axis=-1)

        st.write(
            f"Shape del volumen después de preprocess_volume: {resized_volume_4d.shape}"
        )  # Agrega esta línea

        for i in range(modalities):
            non_zero_mask = resized_volume_4d[..., i] > 0
            mean = np.mean(resized_volume_4d[..., i][non_zero_mask])
            std = np.std(resized_volume_4d[..., i][non_zero_mask])
            resized_volume_4d[..., i][
                non_zero_mask
            ] = (resized_volume_4d[..., i][non_zero_mask] - mean) / std

        return resized_volume_4d
    else:
        st.error("El volumen no tiene el número esperado de dimensiones (4D).")
        return None

def plot_mri_slices(data, modality, overlay=None):
    st.subheader(f"{modality} MRI")

    slice_idx = st.slider(
        f"Selecciona un corte axial para {modality}",
        0,
        data.shape[2] - 1,
        data.shape[2] // 2,
    )
    fig, ax = plt.subplots()
    ax.imshow(data[:, :, slice_idx], cmap="gray")

    # Mostrar overlay si se proporciona
    if overlay is not None:
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
        model = UNet(n_channels=4, n_classes=3)  # Instancia el modelo
        st.write(f"Intentando cargar el modelo desde {MODEL_PATH}...")
        state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        st.success("Modelo cargado correctamente.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.write(traceback.format_exc())  # Imprime el traceback en caso de error
    return None


# --- Lógica principal de la aplicación ---
if __name__ == "__main__":
    # Intenta cargar el modelo al iniciar
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

    # Página de visualización MRI
    if pagina == "Visualización MRI":
        st.title("Visualización de MRI")
        st.write(
            "Sube los archivos NIfTI de diferentes modalidades para visualizar los cortes."
        )

        uploaded_files = st.file_uploader(
            "Sube los archivos MRI (T1, T2, T1c, FLAIR) en formato NIfTI",
            type=["nii", "nii.gz"],
            accept_multiple_files=True,  # Permite subir múltiples archivos
        )

        if uploaded_files:
            # Combinar los archivos en un volumen 4D
            img_data = np.stack([load_nifti(file) for file in uploaded_files], axis=-1)
            if img_data is not None:
                plot_mri_slices(img_data, "MRI Multimodalidad")

    # Página de Resultados de Segmentación
    elif pagina == "Resultados de Segmentación":
        st.title("Resultados de Segmentación")
        st.write(
            "Aquí se mostrarán los resultados de la segmentación del tumor. Sube el archivo MRI para segmentar."
        )

        uploaded_file = st.file_uploader(
            "Sube el archivo MRI apilado (T1, T2, FLAIR, etc.) en formato NIfTI",
            type=["nii", "nii.gz"],
        )

        if uploaded_file is not None:
            img_data = load_nifti(uploaded_file)
            if img_data is not None:
                st.write("Imagen cargada correctamente.")

                try:
                    # Preprocesamiento
                    img_preprocessed = preprocess_volume(img_data)
                    st.write(
                        f"Shape después de preprocess_volume: {img_preprocessed.shape}"
                    ) 

                    if img_preprocessed is not None and model is not None:
                        st.write("Realizando la segmentación...")
                        with torch.no_grad():
                            img_tensor = torch.tensor(img_preprocessed).unsqueeze(0).float()
                            pred = model(img_tensor) 
                            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
                        # Visualización
                        plot_mri_slices(img_preprocessed[..., 0], "MRI Original", overlay=pred)

                except Exception as e:
                    st.error(f"Error durante la segmentación: {e}")
                    st.write(traceback.format_exc()) 
            else:
                st.error("No se pudo cargar la imagen para la segmentación.")

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
        # (Código igual que antes) ...
        pass 

    # --- Página sobre Planificación Quirúrgica ---
    elif pagina == "Planificación Quirúrgica":
       # (Código igual que antes) ...
        pass

    # --- Mensaje de pie de página ---
    st.sidebar.markdown("---")
    st.sidebar.info("Desarrollado por el Grupo 1 de ACSI")
