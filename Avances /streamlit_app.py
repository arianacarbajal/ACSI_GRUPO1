import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Crear las páginas en la barra lateral
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a", ["Visualización MRI", "Resultados de Segmentación", "Leyendas", "Manual de Usuario", "Planificación Quirúrgica"])

# Función para cargar y mostrar archivos MRI
def load_nifti(file):
    if file is not None:
        file_bytes = BytesIO(file.read())
        img = nib.load(file_bytes)
        return img.get_fdata()
    return None

# Función para mostrar cortes de las imágenes
def plot_mri_slices(data, modality):
    st.subheader(f"{modality} MRI")
    slice_idx = st.slider(f"Selecciona un corte axial para {modality}", 0, data.shape[2] - 1, data.shape[2] // 2)
    plt.imshow(data[:, :, slice_idx], cmap='gray')
    plt.axis('off')
    st.pyplot(plt)

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
    
    seg_file = st.file_uploader("Sube el archivo de segmentación (NIfTI)", type=["nii", "nii.gz"])
    
    if seg_file:
        seg_data = load_nifti(seg_file)
        if seg_data is not None:
            st.write("Segmentación cargada correctamente.")
            unique_values = np.unique(seg_data)
            st.write(f"Valores únicos en la segmentación (etiquetas de los tejidos): {unique_values}")
            
            # Mostrar una imagen de segmentación
            slice_idx = st.slider("Selecciona un corte axial para visualizar", 0, seg_data.shape[2] - 1, seg_data.shape[2] // 2)
            plt.imshow(seg_data[:, :, slice_idx], cmap='gray')
            plt.axis('off')
            st.pyplot(plt)

# Página de Leyendas
elif pagina == "Leyendas":
    st.title("Leyendas de Segmentación")
    st.write("""
    En las imágenes segmentadas, cada valor representa un tipo de tejido. A continuación se muestra la leyenda para interpretar las imágenes:

    - **0**: Fondo
    - **1**: Núcleo de tumor necrótico (azul)
    - **2**: Tumor realzado (amarillo)
    - **3**: Tejido edematoso peritumoral (verde)
    """)

# Página del Manual de Usuario
elif pagina == "Manual de Usuario":
    st.title("Manual de Usuario")
    st.write("""
    **Manual de Uso del Visualizador de MRI:**

    1. **Cargar Archivos**: Sube los archivos MRI en formato NIfTI para cada modalidad (T1, T2, T1c, FLAIR).
    2. **Visualización de Cortes**: Usa el control deslizante para seleccionar el corte axial que desees visualizar.
    3. **Segmentación**: Sube el archivo de segmentación para visualizar las etiquetas correspondientes al tumor.
    """)

# Página sobre Planificación Quirúrgica
elif pagina == "Planificación Quirúrgica":
    st.title("Aplicaciones en la Planificación Quirúrgica")
    st.write("""
    La segmentación de imágenes cerebrales juega un papel crucial en la planificación de cirugías para la resección de tumores cerebrales. 
    Al identificar el núcleo del tumor, el tejido edematoso y el tumor realzado, los cirujanos pueden planificar estrategias precisas para la intervención quirúrgica.

    Este sistema de visualización y segmentación permite a los médicos observar la estructura del tumor en detalle y tomar decisiones informadas sobre la ruta quirúrgica y el tratamiento más adecuado.
    """)
