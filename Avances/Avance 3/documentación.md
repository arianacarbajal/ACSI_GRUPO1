El estudio se centró en la segmentación de gliomas, que son el tipo más común de tumores cerebrales en humanos. Los gliomas tienen formas y apariencias variables en las imágenes de resonancia magnética (MRI) multimodal, lo que hace que su segmentación precisa sea un desafío importante en el análisis de imágenes médicas [1].
Las etiquetas de segmentación utilizadas en el conjunto de datos del desafío BraTS21 incluyeron cuatro clases principales relacionadas con los gliomas:

1. Tumor que mejora con contraste (Enhancing Tumor, ET).
2. Tejido edematoso peritumoral (Peritumoral Edematous Tissue, ED).
3. Núcleo necrótico del tumor (Necrotic Tumor Core, NCR).
4. Fondo (Background), que representa los vóxeles que no forman parte del tumor.

Estas cuatro clases fueron las que se utilizaron para entrenar el modelo de segmentación en el estudio, basándose en imágenes multimodales de resonancia magnética de los pacientes con gliomas.

[1] Mayo Clinic, "Glioma: Symptoms and causes", Mayo Clinic, 2023. [En línea]. Disponible en: https://www.mayoclinic.org/es/diseases-conditions/glioma/symptoms-causes/syc-20350251.

## ¿Qué tipos de gliomas existen y cuáles son sus grados?

Los **gliomas** son un tipo de tumor cerebral que se origina en las células gliales del cerebro o la médula espinal. Las células gliales son células de soporte en el sistema nervioso que proporcionan estructura, protección y asistencia a las neuronas. A diferencia de las neuronas, que son responsables de la transmisión de señales, las células gliales desempeñan funciones auxiliares cruciales.

### Tipos de gliomas
Existen varios tipos de gliomas, clasificados según el tipo específico de célula glial de la que se originan:

1. **Astrocitomas**: Son los gliomas más comunes y se desarrollan en los astrocitos, un tipo de célula glial que soporta a las neuronas. Los astrocitomas pueden variar desde tumores de bajo grado (menos agresivos) hasta de alto grado (muy agresivos), como el **glioblastoma multiforme (GBM)**, que es el más maligno.
   
2. **Oligodendrogliomas**: Se originan en las células gliales llamadas oligodendrocitos, que producen mielina, la sustancia que aísla las fibras nerviosas. Estos tumores tienden a crecer más lentamente que otros tipos de gliomas.

3. **Ependimomas**: Se desarrollan en las células ependimarias, que recubren los ventrículos del cerebro y el canal central de la médula espinal. Estos tumores pueden bloquear el flujo de líquido cefalorraquídeo, causando síntomas como dolor de cabeza o hidrocefalia.

### Clasificación por grado
Los gliomas se clasifican en grados (de I a IV) según su agresividad:

- **Grado I y II**: Son gliomas de bajo grado y generalmente crecen más lentamente. Los síntomas pueden ser menos severos y progresar de manera más gradual.
  
- **Grado III y IV**: Son gliomas de alto grado y más agresivos. Los gliomas de grado IV, como el **glioblastoma multiforme (GBM)**, son los más malignos y de peor pronóstico, con un crecimiento rápido y resistencia a muchos tratamientos.

## ¿Qué tipos de Gliomas son más comunes?

El tipo de glioma más frecuente es el glioblastoma multiforme (GBM), que es un tipo de astrocitoma de grado IV, lo que lo convierte en el más agresivo y maligno de los tumores cerebrales. Representa alrededor del 50% de todos los gliomas diagnosticados [2].

[2] OncoLink, "Glioma en adultos: Clasificación y tratamiento", OncoLink, 2018. [En línea]. Disponible en: https://es.oncolink.org/tipos-de-cancer/tumores-del-cerebro/glioma-en-adultos-clasificacion-y-tratamiento.

## ¿En qué se diferencias los otros tipos de tumores cebrebrales con los gliomas?

| **Tipo de Tumor**                            | **Origen**                                                               | **Comportamiento**                                             | **Características**                                                                                         |
|----------------------------------------------|--------------------------------------------------------------------------|---------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **Gliomas**                                  | Células gliales (astrocitos, oligodendrocitos, células ependimarias)      | Pueden ser de bajo o alto grado (agresivos, como el GBM)       | Suelen ser infiltrantes, lo que dificulta su eliminación completa.                                           |
| **Meningiomas**                              | Meninges (capas que recubren el cerebro y la médula espinal)              | Generalmente benignos y de crecimiento lento                   | Comprimen el cerebro en lugar de invadirlo. Facilitan la cirugía en algunos casos.                           |
| **Schwannomas**                              | Células de Schwann (nervios periféricos)                                 | Benignos y de crecimiento lento                                | Afectan principalmente los nervios craneales o periféricos (ej. neurinoma del acústico).                     |
| **Meduloblastomas**                          | Células neuronales inmaduras en el cerebelo                              | Malignos y altamente agresivos                                 | Comunes en niños, con alto potencial de propagación a través del líquido cefalorraquídeo.                    |
| **Craniofaringiomas**                        | Células embrionarias cerca de la glándula pituitaria                     | Benignos                                                       | Afectan el equilibrio hormonal y la visión debido a su proximidad a la hipófisis y el nervio óptico.          |
| **Linfomas primarios del SNC (LPSNC)**       | Células inmunitarias (linfocitos) en el cerebro o la médula espinal       | Malignos y agresivos                                           | No se tratan con cirugía, responden mejor a quimioterapia y radioterapia.                                    |

| **Característica**                              | **Cáncer Metastásico en el Cerebro**                                     | **Tumor de Origen Cerebral (Primario)**                                  |
|-------------------------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|
| **Origen**                                      | Proviene de un cáncer en otra parte del cuerpo que se disemina al cerebro (ej. cáncer de pulmón, mama o melanoma). | Se origina directamente en las células del cerebro o la médula espinal. |
| **Células Tumorales**                           | Las células del tumor son similares a las del cáncer original, fuera del cerebro. | Las células tumorales son del cerebro, como las gliales o neuronales.    |
| **Ejemplos Comunes**                            | Metástasis cerebral de cáncer de pulmón, mama, riñón, melanoma.          | Gliomas (ej. glioblastoma), meningiomas, ependimomas, meduloblastomas.   |
| **Frecuencia**                                  | Es más común que los tumores cerebrales primarios en adultos.            | Menos frecuente que las metástasis cerebrales, pero varía según la población. |
| **Comportamiento**                              | Generalmente se diseminan desde otras áreas del cuerpo al cerebro; pueden afectar múltiples regiones del cerebro. | Crecen localmente en el cerebro, pero algunos pueden invadir el tejido circundante. |

## ¿Cómo se hace un diagnóstico efectivo de los gliomas?

El diagnóstico efectivo de los **gliomas** implica una combinación de **evaluaciones clínicas**, **imágenes cerebrales** y, en algunos casos, **biopsias** y **análisis genéticos**. A continuación se detallan los pasos clave para diagnosticar un glioma:

### 1. Evaluación Clínica
El proceso diagnóstico comienza con:
- **Historia clínica detallada**: Revisión de síntomas como dolores de cabeza, convulsiones, cambios en la personalidad, problemas motores, entre otros.
- **Examen neurológico**: Se evalúan los reflejos, la coordinación, la fuerza muscular, y las funciones cognitivas.

### 2. Imágenes Cerebrales
Las imágenes del cerebro son fundamentales para detectar y caracterizar el tumor.

#### a. Resonancia Magnética (MRI)
- **Técnica de elección** para diagnosticar gliomas.
- **T1 con gadolinio**: Evalúa la extensión del tumor y su vascularización.
- **FLAIR y T2**: Identifican áreas de edema alrededor del tumor.
- **Espectroscopia por RM** o **Perfusión por RM**: Ayudan a analizar el metabolismo y la vascularización del tumor.

#### b. Tomografía Computarizada (TC)
- Menos sensible que la MRI, pero útil para detectar calcificaciones o hemorragias.
- Se utiliza en emergencias cuando no se dispone de una MRI.

#### c. Tomografía por Emisión de Positrones (PET)
- Evalúa la **actividad metabólica** del tumor y diferencia entre tumor activo y tejido cicatricial.

### 3. Biopsia
Para confirmar el diagnóstico, se puede realizar una biopsia del tejido tumoral:

#### a. Biopsia Estereotáctica
- **Minimamente invasiva**. Se extrae una muestra del tumor utilizando una aguja guiada por imágenes tridimensionales.

#### b. Craniectomía o Craniotomía
- **Cirugía abierta** para extirpar el tumor o parte de él. Se analiza el tejido para determinar el tipo y grado del glioma.

### 4. Análisis Genético y Molecular
Las pruebas genéticas y moleculares permiten evaluar mutaciones clave en el tumor:
- **Mutación IDH**: Indica un mejor pronóstico.
- **Codeleción 1p/19q**: Asociada con oligodendrogliomas y mejor respuesta a tratamientos.
- **Mutaciones TP53 y ATRX**: Indicativas de ciertos astrocitomas.

## ¿Cómo se sabe que una segmentación manual tiene margen de error?


## ¿Por qué en el paper toman esos tipos de contrastes, existen más?

En el paper mencionan el uso de diferentes secuencias de resonancia magnética (MRI) para la segmentación de tumores cerebrales, específicamente las secuencias T1, T2, T1 post-contraste (T1Gd) y FLAIR. Estas secuencias se seleccionan porque permiten observar diferentes características del tumor y del tejido cerebral circundante. Cada secuencia destaca diferentes aspectos anatómicos y patológicos, y la combinación de estas proporciona una visión más completa de los tumores cerebrales.

* T1:  Destaca la anatomía del cerebro. Proporciona una imagen clara de las estructuras normales del cerebro y de las lesiones con contraste, pero el líquido cefalorraquídeo (LCR) se muestra oscuro. Después de la administración de gadolinio (T1 con gadolinio o T1Gd), es excelente para ver la mejora del contraste en los tumores, que suele ser un signo de malignidad.
* T2 ponderada: Muestra los líquidos del cerebro como brillantes. Es útil para detectar edema y áreas afectadas por el tumor.
* FLAIR: Similar a T2, pero con supresión de señal del líquido cefalorraquídeo (LCR), haciendo que las áreas patológicas sean más visibles. Es especialmente útil para detectar lesiones cercanas a los ventrículos y diferenciar el edema del líquido normal, lo que mejora la visualización de las áreas infiltradas por el tumor.
* T1 post-contraste (T1Gd): Se administra un contraste de gadolinio para mejorar la imagen T1. Resalta las áreas donde hay una barrera hematoencefálica comprometida, como los tumores. Mejora las áreas del tumor que tienen un aumento en la vascularización o una mayor permeabilidad de los vasos sanguíneos, lo que es común en los tumores malignos.

Existen más pero no se añadieron ya que estas secuencias proporcionan una visión complementaria y eficaz de las características clave de los tumores. Mientras que T1Gd resalta áreas de realce por el contraste (donde la barrera hematoencefálica está comprometida), T2 y FLAIR son excelentes para visualizar el edema y las áreas infiltradas del cerebro. En conjunto, permiten identificar las subregiones del tumor (núcleo tumoral, necrosis, edema).

Usar más secuencias podría no aportar significativamente más información relevante para la segmentación en comparación con el tiempo y los recursos adicionales que requeriría.

También, la inclusión de secuencias adicionales como DWI (Difusión) o espectroscopia por RM podría aumentar la complejidad del procesamiento sin un beneficio claro en la segmentación de gliomas para este contexto específico. Las cuatro secuencias seleccionadas ya cubren las áreas anatómicas y funcionales clave del tumor, haciendo que agregar más secuencias no necesariamente incremente la precisión de la segmentación en los modelos de red neuronal profunda utilizados.

Incluir muchas más secuencias podría aumentar considerablemente el tiempo de adquisición de imágenes, lo que podría no ser viable para estudios clínicos de rutina. Además, en términos de procesamiento computacional, incorporar más secuencias haría que el análisis sea más complejo y más costoso en cuanto a tiempo y recursos de procesamiento.
