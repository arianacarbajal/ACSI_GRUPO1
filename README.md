

# Proyecto ACSI : Optimized U-Net for Brain Tumor Segmentation


## Resumen 

Este estudio propone un sistema en tiempo real de monitoreo de fatiga muscular basado en sensores de electromiografía de superficie (sEMG) y aprendizaje automático para optimizar la rehabilitación y prevenir lesiones musculares por sobreesfuerzo durante ejercicios en bicicleta estática. El sistema se centra específicamente en el músculo vasto lateral por su importante rol en la generación de fuerza durante el pedaleo.

Se desarrolló un modelo de clasificación RandomForest que logró una precisión de 90% en la detección de estados de fatiga y no fatiga muscular durante una prueba incremental de 10 minutos en bicicleta estática. 

Los resultados de la prueba estadística de Wilcoxon confirmaron diferencias significativas en las características de la señal EMG antes y después del umbral de detección de fatiga hallado por el modelo, validando su capacidad para identificar correctamente el inicio de la fatiga.

## Motivación

La falta de monitoreo adecuado de la fatiga muscular durante ejercicios de rehabilitación puede ocasionar sobreesfuerzo, retraso en la recuperación e incluso nuevas lesiones. Por ello, es vital desarrollar sistemas confiables de detección de fatiga en tiempo real para optimizar las rutinas de rehabilitación.

La elección de enfocarse en el ejercicio de bicicleta estática se basó en que es un método idóneo de rehabilitación de lesiones de miembros inferiores, al permitir modular la posición para trabajar grupos musculares específicos. Además, involucra la activación de músculos propensos a lesiones como el vasto lateral, crucial en la generación de fuerza durante el pedaleo.

## Metodología

Se utilizaron electrodos de superficie para adquirir señales EMG del músculo vasto lateral durante una prueba incremental de 10 minutos en bicicleta estática. Las señales fueron preprocesadas con filtrado digital y segmentación.

Se extrajeron características estadísticas y basadas en transformada wavelet para entrenar un modelo RandomForest de clasificación entre estados de fatiga y no fatiga. El conjunto de datos fue dividido 80/20 en entrenamiento y prueba previa normalización de características.

## Principales hallazgos

- El modelo RandomForest alcanzó 90% de precisión en la clasificación de estados de fatiga muscular a partir de las señales sEMG registradas.

- Los coeficientes de wavelet de niveles iniciales demostraron ser indicadores clave de fatiga.

- La prueba de Wilcoxon confirmó diferencias estadísticamente significativas en las características extraídas antes y después del umbral de detección de fatiga.

- El sistema propuesto es prometedor para la implementación en entornos reales de rehabilitación y la optimización de protocolos.

## Conclusiones

El modelo de aprendizaje automático desarrollado permite detectar confiablemente el inicio de la fatiga muscular durante ejercicios de rehabilitación en bicicleta estática a partir del análisis de señales sEMG. Esto facilitaría el monitoreo en tiempo real para ajustar oportunamente las cargas de trabajo y prevenir sobreesfuerzo y complicaciones.

Se sentaron bases para investigaciones orientadas a evaluar el impacto de este tipo de sistemas de monitoreo inteligente de fatiga muscular en los resultados a mediano y largo plazo de programas de rehabilitación física.

### Integrantes del proyecto

- Ariana Carbajal (colaborador) - ariana.carbajal@upch.pe 
- Natalia Galindo Concha (colaborador) - natalia.galindo@upch.pe 
- Gianfranco Fabian Feria Maquera (colaborador) - gianfranco.feria@upch.pe 
- Eduardo André Cuti Riveros (colaborador) -  eduardo.cuti@upch.pe 


Para mayor información sobre el equipo de investigación, revisar el [Entregable 1][enlace-entregable1]

[enlace-entregable1]: https://github.com/arianacarbajal/ISB_Grupo3/blob/3df8663a70e21ea88d1cc997fa5a18c9b391c73e/ISB/Laboratorios/1.Sobre%20nosotros.md


Para mayor información sobre el código empleado , revisar el [Código Entregable 1][enlace-entregable1]

[enlace-entregable1]:https://github.com/arianacarbajal/ISB_Grupo3/tree/eedd6beeb8655bfa58397a819f8e53da771458a6/Software
