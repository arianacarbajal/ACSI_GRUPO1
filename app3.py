import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
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
            temp_file.write(file.read())  # Escribir el archivo en el sistema temporal
            temp_file.flush()  # Asegurarse de que el archivo esté completamente escrito
            img = nib.load(temp_file.name)  # Cargar el archivo NIfTI desde el archivo temporal
            return img.get_fdata()
    return None

def load_nifti(file):
    if file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_file:
                temp_file.write(file.read())
                temp_file.flush()
                img = nib.load(temp_file.name)
                return img.get_fdata(), img.shape  # Devolver los datos de la imagen y su forma
        except Exception as e:
            st.error(f"Error al cargar el archivo NIfTI: {str(e)}")
    return None, None


def preprocess_volume(volume, target_shape=(128, 128)):
    """
    Preprocesa un volumen 4D (o 3D si se trata de un solo canal) para que sea compatible con
    el modelo U-Net 2D.

    Args:
        volume (np.array): El volumen a preprocesar. Puede ser 3D (alto, ancho, profundidad)
                           o 4D (alto, ancho, profundidad, canales).
        target_shape (tuple): La forma deseada para las dimensiones alto y ancho
                             después del preprocesamiento (por defecto: (128, 128)).

    Returns:
        np.array: El volumen preprocesado con forma (alto, ancho, profundidad)
                  o (alto, ancho, profundidad, canales), dependiendo del volumen de entrada.

    """
    # Constantes para el recorte del volumen (locales a la función)
    START_SLICE = 40
    END_SLICE = 130
    st.write(f"Verificando las dimensiones del volumen cargado: {volume.shape}")

    # 1. Recortar la profundidad (si es necesario)
    volume = volume[:, :, START_SLICE:END_SLICE]

    # 2. Redimensionar las dimensiones espaciales (si es necesario)
    if volume.shape[0] != target_shape[0] or volume.shape[1] != target_shape[1]:
        st.write("Redimensionando volumen...")
        factors = (target_shape[0] / volume.shape[0],
                   target_shape[1] / volume.shape[1],
                   1)  # Factor 1 para mantener la profundidad
        if len(volume.shape) == 4:  # Si es 4D, redimensionamos cada canal
            new_shape = (target_shape[0], target_shape[1], volume.shape[2], volume.shape[3])
            volume = zoom(volume, factors + (1,),
                          order=1)  # Interpolación lineal
        else:
            volume = zoom(volume, factors, order=1)
        st.write(f"Nuevo tamaño del volumen después de redimensionar: {volume.shape}")
    else:
        st.write("El volumen ya tiene la forma deseada. No se redimensionará.")

    # 3. Normalizar (importante para el entrenamiento, pero opcional para la visualización)
    # Nota: La normalización debe ser consistente con la que utilizaste durante el entrenamiento
    # Ajusta el método de normalización a tus necesidades
    volume = (volume - volume.min()) / (volume.max() - volume.min())

    st.write(f"Shape del volumen después de preprocess_volume: {volume.shape}")
    return volume


def resize_volume_to_shape(volume, target_shape):
    """Redimensiona un volumen a la forma deseada."""
    from scipy.ndimage import zoom
    factors = [target_dim / float(dim) for target_dim, dim in zip(
        target_shape, volume.shape)]
    return zoom(volume, factors, order=1)  # Interpolación lineal


def plot_mri_slices(data, modality, overlay=None):
    """Muestra cortes axiales de un volumen 3D, con la posibilidad de una superposición."""
    st.subheader(f"{modality} MRI")

    if len(data.shape) < 3:
        st.error(
            f"Error: Se esperaban al menos 3 dimensiones en los datos de imagen, pero se encontraron {len(data.shape)}")
        return

    slice_idx = st.slider(
        f"Selecciona un corte axial para {modality}",
        0,
        data.shape[2] - 1,
        data.shape[2] // 2,
    )

    fig, ax = plt.subplots()
    ax.imshow(data[:, :, slice_idx], cmap="gray")  # Mostrar la imagen base en escala de grises

    if overlay is not None:
        # --- Corrección en el manejo del overlay ---
        if overlay.shape[1:] != data.shape[:2]:  # Comparamos (alto, ancho)
            st.error(
                f"Error: Las formas de la imagen y la máscara no coinciden: {data.shape} vs {overlay.shape}")
            return
        else:
            # Mostrar el canal 0 de 'overlay' usando el índice correcto
            ax.imshow(overlay[0, :, :], cmap="hot", alpha=0.6)

    ax.axis("off")
    st.pyplot(fig)

# Función para mostrar cortes de las imágenes
def plot_mri_slices1(data, modality):
    st.subheader(f"{modality} MRI")
    slice_idx = st.slider(f"Selecciona un corte axial para {modality}", 0, data.shape[2] - 1, data.shape[2] // 2)
    plt.imshow(data[:, :, slice_idx], cmap='gray')
    plt.axis('off')
    st.pyplot(plt)

@st.cache_resource
def load_model():
    st.write("Cargando el modelo...")
    if not os.path.exists(MODEL_PATH):
        st.error(f"El archivo del modelo '{MODEL_PATH}' no existe. Descargando...")
        download_model_from_gdrive(MODEL_ID, MODEL_PATH)

    try:
        # Carga el modelo directamente (incluyendo la arquitectura)
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.eval()
        st.success("Modelo cargado correctamente.")
        return model

    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.write(traceback.format_exc())
        return None


# --- Lógica principal de la aplicación ---
if __name__ == "__main__":

    model = load_model()  # Carga el modelo al iniciar la aplicación

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
    # Página de visualización MRI
    if pagina == "Visualización MRI":
        st.title("Visualización de MRI")
        st.write(
            "Sube los archivos NIfTI de diferentes modalidades para visualizar los cortes.")

        t1_file = st.file_uploader(
            "Sube el archivo T1-weighted (T1)", type=["nii", "nii.gz"])
        t1c_file = st.file_uploader(
            "Sube el archivo T1 con contraste (T1c)", type=["nii", "nii.gz"])
        t2_file = st.file_uploader(
            "Sube el archivo T2-weighted (T2)", type=["nii", "nii.gz"])
        flair_file = st.file_uploader(
            "Sube el archivo T2-FLAIR", type=["nii", "nii.gz"])

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

    # --- Sección "Resultados de Segmentación" ---
    elif pagina == "Resultados de Segmentación":
        st.title("Resultados de Segmentación")
        st.write(
            "Aquí se mostrarán los resultados de la segmentación del tumor. Sube el archivo apilado (stack) para segmentar.")

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
                    st.error(
                        "Tipo de archivo no soportado. Por favor, carga un archivo .npy o .nii/.nii.gz.")
                    st.stop()  # Detiene la ejecución si el archivo no es válido

                # Comprobaciones de dimensiones
                if len(img_data.shape) != 4:
                    raise ValueError(
                        f"Error: Se esperaban 4 dimensiones (alto, ancho, profundidad, canales). Se obtuvieron: {img_data.shape}")

                # Preprocesar volumen
                img_preprocessed = preprocess_volume(img_data)

                if img_preprocessed is not None and model is not None:
                    with torch.no_grad():
                        # --- Ajustar dimensiones para el modelo ---
                        img_tensor = torch.tensor(
                            img_preprocessed).unsqueeze(0).float()
                        img_tensor = img_tensor.permute(
                            0, 3, 1, 2)

                        # --- Inferencia ---
                        pred = model(img_tensor)

                        # --- Procesar la salida del modelo ---
                        pred = torch.sigmoid(
                            pred).squeeze(0).cpu().numpy()

                    # --- Visualización ---
                    plot_mri_slices(img_preprocessed[:,
                                    :, :, 0], "T1 Original", overlay=pred)

            except Exception as e:
                st.error(f"Error durante la segmentación: {e}")
                st.write(traceback.format_exc())

elif pagina == "Leyendas":
    st.title("Leyendas de Segmentación")
    st.write(
        """
    En las imágenes segmentadas, cada color representa un tipo de tejido. 
    A continuación se muestra la leyenda para interpretar las imágenes:

    - **Rojo:** Núcleo del tumor necrótico. 
    - **Amarillo:** Tumor realzado (tejido tumoral activo).
    - **Verde:** Tejido edematoso peritumoral (inflamación alrededor del tumor).
    - **Gris oscuro (fondo):**  Tejido cerebral normal. 
    """
    )

elif pagina == "Manual de Usuario":
    st.title("Manual de Usuario")
    st.write(
        """
    **Cómo utilizar esta aplicación:**

    **1. Visualización de Imágenes MRI:**
      * En la sección "Visualización MRI", sube archivos de imágenes cerebrales en formato NIfTI (.nii o .nii.gz). 
      * Puedes subir archivos para las siguientes modalidades: T1, T1c (con contraste), T2 y FLAIR. 
      * Utiliza el control deslizante para visualizar diferentes cortes axiales de la imagen.

    **2. Segmentación de Tumor Cerebral:**
      * En la sección "Resultados de Segmentación", sube un archivo que contenga las cuatro modalidades (T1, T1c, T2, FLAIR) en un solo archivo (formato .npy o .nii/.nii.gz). 
      * El modelo de aprendizaje profundo segmentará automáticamente la imagen y mostrará la región del tumor. 
      * Utiliza el control deslizante para visualizar la segmentación en diferentes cortes axiales. 

    **3. Interpretación de Resultados:**
      * Revisa la página "Leyendas" para entender el significado de los colores en la segmentación. 

    **4. Consideraciones importantes:**
      * Asegúrate de que los archivos de imagen que subes sean de buena calidad para obtener resultados óptimos. 
      * Este sistema es una herramienta de apoyo y no debe utilizarse como único medio para el diagnóstico o la planificación del tratamiento.
    """
    )

elif pagina == "Planificación Quirúrgica":
    st.title("Aplicaciones en la Planificación Quirúrgica")
    st.write(
        """
    La segmentación precisa de tumores cerebrales es esencial para la planificación quirúrgica, ya que proporciona información crucial para:

    * **Delimitación del tumor:**  Definir con precisión los límites del tumor, incluyendo el núcleo necrótico, la región realzada y el área edematosa.
    * **Planificación de la resección:**  Ayudar a los cirujanos a planificar la mejor estrategia quirúrgica para la extirpación del tumor, teniendo en cuenta su ubicación y proximidad a estructuras cerebrales críticas.
    * **Minimización de riesgos:**  Evaluar la probabilidad de dañar tejidos sanos durante la cirugía y minimizar el riesgo de complicaciones.
    * **Selección de la ruta de acceso:**  Determinar la mejor manera de acceder al tumor durante la cirugía.

    **Beneficios de la segmentación para la planificación quirúrgica:**

    * **Cirugía más precisa y segura.**
    * **Reducción del riesgo de complicaciones.**
    * **Mejores resultados para los pacientes.**
    * **Toma de decisiones informadas.**

    **Es importante tener en cuenta que:**

    * Esta aplicación proporciona una segmentación del tumor, pero no reemplaza la evaluación y el juicio clínico de un profesional médico. 
    * La planificación quirúrgica final debe basarse en la combinación de la segmentación, imágenes médicas adicionales y la experiencia del equipo quirúrgico.
    """
    )

    # --- Mensaje de pie de página ---
    st.sidebar.markdown("---")
    st.sidebar.info("Desarrollado por el Grupo 1 de ACSI")
