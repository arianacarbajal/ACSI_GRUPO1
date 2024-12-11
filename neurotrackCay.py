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
from PIL import Image

# --- Configuración de la página ---
st.set_page_config(page_title="NeuroTrackCayetano", layout="wide")

# ---  Configuración del modelo ---
MODEL_ID = '1qb0pRpbX9PrZK5yRRAXU_N_0x_gLsWzq' 
MODEL_PATH = 'trained_model_final.pth' # Reemplaza con la ruta de tu modelo

# --- Definición del modelo U-Net 2D ---
class DoubleConv(nn.Module):
    """(convolución => [BN] => ReLU) * 2"""

    def __init__(self, canales_entrada, canales_salida):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(canales_entrada, canales_salida, kernel_size=3, padding=1),
            nn.BatchNorm2d(canales_salida),
            nn.ReLU(inplace=True),
            nn.Conv2d(canales_salida, canales_salida, kernel_size=3, padding=1),
            nn.BatchNorm2d(canales_salida),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Reducción de escala con maxpool y luego doble convolución"""

    def __init__(self, canales_entrada, canales_salida):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(canales_entrada, canales_salida)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Aumento de escala y luego doble convolución"""

    def __init__(self, canales_entrada, canales_salida, bilinear=True):
        super().__init__()

        # Usar muestreo ascendente bilineal o convolución transpuesta
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(canales_entrada // 2, canales_entrada // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(canales_entrada, canales_salida)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # La entrada es CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, canales_entrada=4, num_clases=3, bilinear=True):
        super(UNet, self).__init__()
        self.canales_entrada = canales_entrada
        self.num_clases = num_clases
        self.bilinear = bilinear

        self.inc = DoubleConv(canales_entrada, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, num_clases, kernel_size=1)

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
def descargar_modelo_desde_gdrive(model_id, ruta_modelo):
    try:
        gdown.download(f'https://drive.google.com/uc?id={model_id}', ruta_modelo, quiet=True)
    except Exception as e:
        # Opcionalmente puedes registrar el error en un archivo log o simplemente ignorarlo
        pass


def cargar_nifti1(archivo):
    if archivo is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as archivo_temporal:
            archivo_temporal.write(archivo.read())  # Escribir el archivo en el sistema temporal
            archivo_temporal.flush()  # Asegurarse de que el archivo esté completamente escrito
            img = nib.load(archivo_temporal.name)  # Cargar el archivo NIfTI desde el archivo temporal
            return img.get_fdata()
    return None

# Función para mostrar cortes de las imágenes
def mostrar_cortes_mri1(datos, modalidad):
    st.subheader(f"MRI {modalidad}")
    indice_corte = st.slider(f"Selecciona un corte axial para {modalidad}", 0, datos.shape[2] - 1, datos.shape[2] // 2)
    plt.imshow(datos[:, :, indice_corte], cmap='gray')
    plt.axis('off')
    st.pyplot(plt)

def preprocesar_volumen(volumen, forma_objetivo=(128, 128)):
    """
    Preprocesa un volumen 4D (o 3D si se trata de un solo canal) para que sea compatible co
    el modelo U-Net 2D.

    Args:
        volumen (np.array): El volumen a preprocesar. Puede ser 3D (alto, ancho, profundidad)
                           o 4D (alto, ancho, profundidad, canales).
        forma_objetivo (tuple): La forma deseada para las dimensiones alto y ancho
                             después del preprocesamiento (por defecto: (128, 128)).

    Returns:
        np.array: El volumen preprocesado con forma (alto, ancho, profundidad)
                  o (alto, ancho, profundidad, canales), dependiendo del volumen de entrada.
    """

    st.write(f"Verificando las dimensiones del volumen cargado: {volumen.shape}")

    # 1. Recortar la profundidad (si es necesario)
    CORTE_INICIAL = 0
    CORTE_FINAL = 155  
    volumen = volumen[:, :, CORTE_INICIAL:CORTE_FINAL]

    # 2. Redimensionar las dimensiones espaciales (si es necesario)
    if volumen.shape[0] != forma_objetivo[0] or volumen.shape[1] != forma_objetivo[1]:
        st.write("Redimensionando volumen...")
        factores = (
            forma_objetivo[0] / volumen.shape[0],
            forma_objetivo[1] / volumen.shape[1],
            1,  # Factor 1 para mantener la profundidad
        )
        if len(volumen.shape) == 4:  # Si es 4D, redimensionamos cada canal
            nueva_forma = (forma_objetivo[0], forma_objetivo[1], volumen.shape[2], volumen.shape[3])
            volumen = zoom(volumen, factores + (1,), order=1)  # Interpolación lineal
        else:
            volumen = zoom(volumen, factores, order=1)
        st.write(f"Nuevo tamaño del volumen después de redimensionar: {volumen.shape}")
    else:
        st.write("El volumen ya tiene la forma deseada. No se redimensionará.")

    # 3. Normalizar
    volumen = (volumen - volumen.min()) / (volumen.max() - volumen.min())

    st.write(f"Shape del volumen después de preprocesar_volumen: {volumen.shape}")
    return volumen

def mostrar_cortes_mri(datos, modalidad, superposicion=None):
    """Muestra cortes axiales de un volumen 3D, con la posibilidad de una superposición."""
    st.subheader(f"MRI {modalidad}")

    if len(datos.shape) < 3:
        st.error(f"Error: Se esperaban al menos 3 dimensiones en los datos de imagen, pero se encontraron {len(datos.shape)}")
        return

    indice_corte = st.slider(
        f"Selecciona un corte axial para {modalidad}",
        0,
        datos.shape[2] - 1,
        datos.shape[2] // 2,
    )

    fig, ax = plt.subplots()
    ax.imshow(datos[:, :, indice_corte], cmap="gray")  # Mostrar la imagen base en escala de grises

    if superposicion is not None:
        # --- Corrección en el manejo de la superposición ---
        if superposicion.shape[1:] != datos.shape[:2]:  # Comparamos (alto, ancho)
            st.error(f"Error: Las formas de la imagen y la máscara no coinciden: {datos.shape} vs {superposicion.shape}")
            return
        else:
            # Mostrar el canal 0 de 'superposicion' usando el índice correcto
            ax.imshow(superposicion[0, :, :], cmap="hot", alpha=0.6)

    ax.axis("off")
    st.pyplot(fig)

@st.cache_resource
def cargar_modelo():
    try:
        # Intentamos cargar el modelo desde el archivo, especificando que se cargue en la CPU
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        # Extraer solo el 'model_state_dict'
        model_state_dict = checkpoint['model_state_dict']
        
        # Crear una instancia del modelo U-Net
        modelo = UNet()
        
        # Cargar el estado del modelo en la instancia
        modelo.load_state_dict(model_state_dict)
        
        modelo.eval()  # Establecer el modelo en modo evaluación
        st.write("Modelo cargado correctamente.")
        
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.stop()  # Detener la ejecución si el modelo no se carga correctamente


# --- Lógica principal de la aplicación ---
if __name__ == "__main__":

    modelo = cargar_modelo()

    # Barra lateral
    st.sidebar.title("Navegación")
    pagina = st.sidebar.radio(
        "Ir a",
        [
            "Inicio",
            "Visualización MRI",
            "Resultados de Segmentación",
            "Leyendas",
            "Manual de Usuario",
            "Planificación Quirúrgica",
        ],
    )

    # --- Página de Visualización MRI ---
    if pagina == "Inicio":
        st.title("NeuroTrackCayetano: Software de Planeación Quirúrgica Basada en Segmentación de Tumores Cerebrales")
        st.write(""" Bienvenido a **NeuroTrackCayetano**, una herramienta avanzada diseñada para apoyar a los neurocirujanos
        en la planificación quirúrgica de tumores cerebrales mediante la segmentación precisa de imágenes
        de resonancia magnética (MRI).""")
        
        st.write("""
        ### Propósito del Sistema
        NeuroTrackCayetano utiliza algoritmos de segmentación basados en aprendizaje profundo 
        para identificar y diferenciar regiones clave dentro del cerebro afectadas por tumores.
        Estas regiones incluyen:
        - **Núcleo necrótico del tumor**.
        - **Tejido tumoral realzado**.
        - **Zona edematosa peritumoral**.
        """)
        ruta_imagen = "ejemplo.png"  
        st.image(ruta_imagen, caption="Representación Visual de los Resultados del Análisis con NeuroTrackCayetano", use_column_width=True)             
        st.write("""Explora las diferentes secciones del sistema usando la barra de navegación a la izquierda. ¡Comencemos!""")
    elif pagina == "Visualización MRI":
        st.title("Visualización de Imágenes MRI")
        st.write("Sube los archivos NIfTI de diferentes modalidades para visualizar los cortes.")
        archivo_t1 = st.file_uploader("Sube el archivo T1-weighted (T1)", type=["nii", "nii.gz"])
        archivo_t1c = st.file_uploader("Sube el archivo T1 con contraste (T1c)", type=["nii", "nii.gz"])
        archivo_t2 = st.file_uploader("Sube el archivo T2-weighted (T2)", type=["nii", "nii.gz"])
        archivo_flair = st.file_uploader("Sube el archivo T2-FLAIR", type=["nii", "nii.gz"])
        if archivo_t1:
            datos_t1 = cargar_nifti1(archivo_t1)
            if datos_t1 is not None:
                mostrar_cortes_mri1(datos_t1, "T1-weighted")

        if archivo_t1c:
            datos_t1c = cargar_nifti1(archivo_t1c)
            if datos_t1c is not None:
                mostrar_cortes_mri1(datos_t1c, "T1c (con contraste)")

        if archivo_t2:
            datos_t2 = cargar_nifti1(archivo_t2)
            if datos_t2 is not None:
                mostrar_cortes_mri1(datos_t2, "T2-weighted")

        if archivo_flair:
            datos_flair = cargar_nifti1(archivo_flair)
            if datos_flair is not None:
                mostrar_cortes_mri1(datos_flair, "T2-FLAIR")
    # --- Sección "Resultados de Segmentación" ---
    elif pagina == "Resultados de Segmentación":
        st.title("Resultados de Segmentación")
        st.write("Aquí se mostrarán los resultados de la segmentación del tumor. Sube el archivo apilado (stack) para segmentar.")

        pila_cargada = st.file_uploader(
            "Sube el archivo apilado de MRI (.npy o .nii/.nii.gz)",
            type=["npy", "nii", "nii.gz"]
        )

        if pila_cargada is not None:
            try:
                # Cargar datos
                if pila_cargada.name.endswith('.npy'):
                    datos_imagen = np.load(pila_cargada)
                elif pila_cargada.name.endswith(('.nii', '.nii.gz')):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as archivo_temporal:
                        archivo_temporal.write(pila_cargada.read())
                        archivo_temporal.flush()
                        imagen_nii = nib.load(archivo_temporal.name)
                        datos_imagen = imagen_nii.get_fdata()
                        st.write("Archivo NIfTI cargado correctamente.")
                    os.remove(archivo_temporal.name)
                else:
                    st.error("Tipo de archivo no soportado. Por favor, carga un archivo .npy o .nii/.nii.gz.")
                    st.stop()

                # Comprobaciones de dimensiones
                if len(datos_imagen.shape) != 4:
                    raise ValueError(f"Error: Se esperaban 4 dimensiones (alto, ancho, profundidad, canales). Se obtuvieron: {datos_imagen.shape}")

                # Preprocesar volumen
                imagen_preprocesada = preprocesar_volumen(datos_imagen)

                if imagen_preprocesada is not None and modelo is not None:
                    # --- Control deslizante ---
                    indice_corte = st.slider(
                        "Selecciona un corte axial para segmentar",
                        0,
                        imagen_preprocesada.shape[2] - 1,
                        imagen_preprocesada.shape[2] // 2,
                    )

                    with torch.no_grad():
                        # --- Seleccionar el corte ---
                        corte_imagen = imagen_preprocesada[:, :, indice_corte, :]

                        # --- Inferencia para un solo corte ---
                        tensor_imagen = torch.tensor(corte_imagen).unsqueeze(0).float()
                        tensor_imagen = tensor_imagen.permute(0, 3, 1, 2) 
                        prediccion = modelo(tensor_imagen)

                        # --- Procesar 'pred' ---
                        prediccion = torch.sigmoid(prediccion).squeeze(0).cpu().numpy()

                    # --- Visualización ---
                    mostrar_cortes_mri(imagen_preprocesada[:, :, :, 0], "T1 Original", superposicion=prediccion)

            except Exception as e:
                st.error(f"Error durante la segmentación: {e}")
                st.write(traceback.format_exc())

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
            - Para visualizar: Sube los archivos MRI en formato NIfTI para cada modalidad (T1, T2, T1c, FLAIR) en la página "Visualización MRI". Puedes subir un único archivo que contenga todas las modalidades o cada modalidad por separado. 
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
