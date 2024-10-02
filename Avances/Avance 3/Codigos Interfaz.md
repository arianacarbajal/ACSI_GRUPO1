#CODIGOS STREAMLIT 
import streamlit as st
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import tempfile

# Título y descripción
st.title("Brain Tumor Segmentation App")
st.write("Sube los archivos NIfTI de diferentes modalidades para observar y segmentar el tumor cerebral en diferentes vistas.")

# Cargar los 4 tipos de imágenes MRI
st.header("Carga los archivos NIfTI MRI correspondientes")
t1_file = st.file_uploader("Sube el archivo T1-weighted (T1)", type=["nii", "nii.gz"])
t1c_file = st.file_uploader("Sube el archivo T1 con contraste (T1c)", type=["nii", "nii.gz"])
t2_file = st.file_uploader("Sube el archivo T2-weighted (T2)", type=["nii", "nii.gz"])
flair_file = st.file_uploader("Sube el archivo T2-FLAIR", type=["nii", "nii.gz"])

# Función para cargar archivos guardándolos temporalmente y leerlos con nibabel
def load_nifti(file):
    if file is not None:
        # Crear un archivo temporal para almacenar el archivo subido
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())  # Guardar los datos en el archivo temporal
            tmp_file_path = tmp_file.name  # Obtener la ruta del archivo temporal

        # Cargar el archivo temporal con nibabel
        img = nib.load(tmp_file_path)
        return img.get_fdata()
    return None

# Función para mostrar cortes de las imágenes
def plot_mri_slices(data, modality):
    st.subheader(f"{modality} MRI")
    slice_idx = st.slider(f"Selecciona un corte axial para {modality}", 0, data.shape[2] - 1, data.shape[2] // 2)
    plt.imshow(data[:, :, slice_idx], cmap='gray')
    plt.axis('off')
    st.pyplot(plt)

# Visualización de los archivos cargados
if t1_file and t1c_file and t2_file and flair_file:
    try:
        # Cargar cada archivo y extraer datos
        t1_data = load_nifti(t1_file)
        t1c_data = load_nifti(t1c_file)
        t2_data = load_nifti(t2_file)
        flair_data = load_nifti(flair_file)

        # Verificar si los archivos fueron cargados correctamente
        if t1_data is not None and t1c_data is not None and t2_data is not None and flair_data is not None:
            st.write("Visualizando los cortes seleccionados de cada modalidad:")
            col1, col2 = st.columns(2)

            with col1:
                plot_mri_slices(t1_data, "T1-weighted")
                plot_mri_slices(t2_data, "T2-weighted")

            with col2:
                plot_mri_slices(t1c_data, "T1c (con contraste)")
                plot_mri_slices(flair_data, "T2-FLAIR")

            # Espacio para mostrar resultados de la segmentación (placeholder)
            st.write("Aquí podrías añadir los resultados del modelo de segmentación de tumores usando nnU-Net o similar.")
        else:
            st.error("No se pudo procesar uno o más archivos. Verifica los formatos.")
    
    except Exception as e:
        st.error(f"Error al procesar las imágenes: {e}")
else:
    st.warning("Por favor, sube todos los archivos MRI para poder visualizarlos.")

#CODIGO INTERFAZ PHYTON 
# Import necessary libraries for plotting
import matplotlib.pyplot as plt

# Create the figure and axis for the ray diagram
fig, ax = plt.subplots(figsize=(10, 6))

# Define positions for the object, lens, and image
lens_position = 0  # Lens is at x = 0
object_position = -150  # Object is 150 mm to the left of the lens
image_position = 300  # Image forms 300 mm to the right of the lens

# Object height (given as 10 mm) and image height (calculated as -20 mm, inverted)
object_height = 10
image_height = -20

# Draw the optical element (lens) as a vertical dashed line at x = 0
ax.axvline(lens_position, color='black', linestyle='--', linewidth=1.5, label="Lens (Optical Element)")

# Draw the object as an arrow at the object position
ax.arrow(object_position, 0, 0, object_height, head_width=10, head_length=5, fc='blue', ec='blue', label="Object")

# Draw the image as an inverted arrow at the image position
ax.arrow(image_position, 0, 0, image_height, head_width=10, head_length=5, fc='red', ec='red', label="Real Image")

# Draw the principal rays
# Ray 1: Parallel to the axis, then refracts through the focal point on the other side
ax.plot([object_position, lens_position, image_position], 
        [object_height, 0, image_height], 'g', label="Ray 1 (Parallel, then through Focal Point)")

# Ray 2: Through the center of the lens (travels straight through)
ax.plot([object_position, image_position], 
        [object_height, image_height], 'orange', label="Ray 2 (Through Lens Center)")

# Ray 3: Through the focal point on the object's side, then emerges parallel to the axis
focal_point_left = lens_position - 100  # Focal point is 100 mm to the left of the lens
ax.plot([object_position, focal_point_left, image_position], 
        [object_height, 0, image_height], 'purple', label="Ray 3 (Through Focal Point, then Parallel)")

# Set plot limits and labels
ax.set_xlim(-200, 400)
ax.set_ylim(-30, 30)
ax.set_xlabel("Position (mm)", fontsize=12)
ax.set_ylabel("Height (mm)", fontsize=12)
ax.set_title("Ray Diagram for a Biconvex Lens", fontsize=14)

# Add a legend and grid
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

# Display the plot
plt.show()

