

# Proyecto ACSI : Optimized U-Net for Brain Tumor Segmentation


## Introducción  

El avance en el procesamiento de imágenes médicas ha revolucionado el diagnóstico y tratamiento de diversas enfermedades, particularmente en la oncología. La segmentación de tumores cerebrales a partir de imágenes de resonancia magnética (MRI) es una tarea crítica que puede influir directamente en las decisiones clínicas. Sin embargo, la segmentación manual es un proceso que consume mucho tiempo, requiere un alto nivel de expertise y es susceptible a errores humanos. La automatización de este proceso mediante el uso de redes neuronales profundas, como la arquitectura U-Net, ha mostrado un gran potencial para superar estas limitaciones, proporcionando resultados más consistentes y rápidos.

## Problemática 

El principal desafío en la segmentación de tumores cerebrales es la variabilidad en la forma y apariencia de los tumores en diferentes pacientes y modalidades de MRI. Además, la segmentación manual realizada por radiólogos es un proceso tedioso, con una alta variabilidad interobservador e intraobservador, lo que puede llevar a inconsistencias en el diagnóstico y tratamiento. El desarrollo de modelos automáticos de segmentación, que igualen o superen el rendimiento humano, es esencial para mejorar la precisión y reproducibilidad del diagnóstico.

El artículo "Optimized U-Net for Brain Tumor Segmentation" aborda esta problemática mediante la optimización de una arquitectura U-Net, una red neuronal convolucional ampliamente utilizada en la segmentación de imágenes médicas. Este proyecto tiene como objetivo replicar y validar los resultados presentados en el artículo, utilizando técnicas de aprendizaje profundo para mejorar la segmentación automática de tumores cerebrales.

## Base de datos

Para este proyecto, se utiliza la base de datos proporcionada por el desafío BraTS21 (Brain Tumor Segmentation Challenge 2021). Esta base de datos es una de las más grandes y completas disponibles para la segmentación de tumores cerebrales, lo que la convierte en un estándar de facto en la investigación en este campo.

- Tipos de Imágenes
Las imágenes en la base de datos BraTS21 son de resonancia magnética (MRI) y cubren cuatro modalidades diferentes:

T1: Imágenes ponderadas en T1, que proporcionan un contraste claro entre la materia gris y blanca.
T1Gd: Imágenes T1 con gadolinio, que realza las áreas con alta vascularidad, como los tumores.
T2: Imágenes ponderadas en T2, que destacan el líquido cerebroespinal y permiten la visualización de edema.
FLAIR (Fluid Attenuated Inversion Recovery): Una modalidad que suprime el líquido cerebroespinal para resaltar las lesiones cerebrales, como el edema asociado con tumores.

- Formatos
Todas las imágenes están almacenadas en formato NIfTI (.nii), que es ampliamente utilizado en neuroimagen. Este formato permite manejar volúmenes de datos 3D, preservando la información espacial necesaria para la segmentación precisa.

- Características del Conjunto de Datos
Resolución: Las imágenes han sido preprocesadas para tener una resolución isotrópica de 1 mm³, con dimensiones de 240x240x155 voxeles.
Etiquetas de Segmentación: Las etiquetas incluyen cuatro clases: tumor realzado (ET), tejido edematoso peritumoral (ED), núcleo necrótico del tumor (NCR), y fondo (voxeles que no son parte del tumor).
Anotación: Las anotaciones de los tumores han sido realizadas manualmente por entre uno y cuatro expertos, lo que asegura una alta calidad en las etiquetas de segmentación.


### Objetivo del proyecto 
El objetivo principal de este proyecto es replicar el modelo optimizado U-Net propuesto en el artículo y evaluar su desempeño en la segmentación de tumores cerebrales utilizando la base de datos BraTS21. Además, se buscará realizar ajustes y mejoras en la arquitectura y el proceso de entrenamiento para adaptar el modelo a las características específicas de las imágenes disponibles.

### Integrantes del proyecto

- Ariana Carbajal (colaborador) - ariana.carbajal@upch.pe 
- Alvaro Sevilla (colaborador) - alvaro.sevilla@upch.pe 
- Claudia Camacho (colaborador) - claudia.camacho@upch.pe 
- Angel Dianderas (colaborador) -  angel.dianderas@upch.pe 

