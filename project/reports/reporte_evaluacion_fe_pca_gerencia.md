# REPORTE DE EVALUACIÓN: FEATURE ENGINEERING & PCA
**De:** Gerencia del Proyecto de Datos F1
**Para:** Equipo de Data Science
**Estado:** CRÍTICO / MALA PRAXIS ESTADÍSTICA

He auditado los notebooks de *Feature Engineering* (`Feature_engineering_v3.ipynb`) y *Reducción de Dimensionalidad* (`PCA_V3.ipynb`). Al igual que con el Clustering, los enfoques metodológicos aquí muestran una desconexión total con las buenas prácticas de la ciencia de datos. Están construyendo castillos de arena estadísticos.

A continuación, detallo los fallos estructurales que hacen que su dataset sea una trampa mortal para cualquier modelo predictivo.

---

## 1. Feature Engineering: La "Fábrica de Ruido"

El encabezado de su propio notebook dice textualmente:
> **Objetivo:** superar 500 variables para PCA significativo

**¿En qué academia les enseñaron que el objetivo de la ingeniería de características es inflar artificialmente el número de variables solo para justificar el uso de PCA?** 
El Análisis de Componentes Principales es una "medicina" para curar la maldición de la dimensionalidad. Ustedes enfermaron al paciente a propósito (creando 540 columnas) solo para poder recetarle la medicina.

### El Desastre Dimensional (Ratio 1:1)
El shape final de su dataset es de **643 eventos (filas) x 540 columnas**. 
Tener casi la misma cantidad de features que de observaciones es el ejemplo de manual de la **maldición de la dimensionalidad**. Ningún algoritmo (y mucho menos uno no supervisado) va a encontrar patrones generalizables aquí; el modelo simplemente va a memorizar el ruido estocástico del dataset.

Han creado bloques enteros de variables altamente colineales y redundantes:
- *Bloque A (stats 3 vueltas)*: 241 columnas.
- *Bloque A2 (slope/CV/range)*: 144 columnas.
- *Bloque B (Ventana extendida 5 y 10 vueltas)*.

Tener la media de 3 vueltas, la media de 5 vueltas, y la tendencia temporal (slope) de la misma métrica (ej. desgaste de neumático) no añade información nueva; solo inyecta multicolinealidad severa.

---

## 2. PCA: Basura Entra, Basura Transformada Sale

Pasaron **492 variables** finales al modelo de PCA tras descartar ridículamente solo 6 columnas por varianza cero.

### La Ilusión de la Retención de Varianza
Ustedes reportan que el Componente 1 (PC1) captura el **26.08%** de la varianza. Y para llegar al 80-90% necesitan 15 componentes. 

¿Saben por qué PC1 captura tanta "varianza"? Porque PCA es un algoritmo lineal que detecta colinealidad. Cuando le lanzan 4 versiones distintas de la misma métrica de velocidad (media de 3 vueltas, media de 5 vueltas, coeficiente de variación, rango), el PCA agrupa esa colinealidad matemática masiva en el primer componente. 
- **Conclusión:** Su PC1 no representa un "Diferencial de Rendimiento táctico" puro, es un basurero de variables duplicadas. La interpretabilidad de estos componentes que presumen en sus reportes es artificial. 

### El Problema de la Imputación Ciega
Al no haber eliminado ninguna columna por alto volumen de nulos (reportan `Cols eliminadas (>50% nulos): 0`), significa que o bien el dataset era mágicamente perfecto (lo dudo en F1 telemetry), o bien imputaron miles de valores con la mediana en columnas de baja densidad, destruyendo la distribución real de los datos antes de estandarizarlos y pasarlos al PCA.

---

## RECOMENDACIONES CRUDAS Y PASOS A SEGUIR

Tienen que detener el pipeline actual. Si los cimientos están podridos, el clustering y los modelos predictivos posteriores nunca funcionarán. 

Deben ejecutar las siguientes acciones hoy:

1. **Purga de Variables (Feature Selection basada en Dominio):**
   Abran su código de Feature Engineering y eliminen el 60% de esas variables inútiles. Si tienen `lap_time_mean_3`, no necesitan `lap_time_mean_5` ni `lap_time_cv`. Quédense con una ventana de tiempo representativa (ej. 3 vueltas) y descarten el resto. Necesitan reducir las dimensiones a un máximo de **50-80 features core** impulsadas por la lógica de F1, no por bucles *for* perezosos.
2. **Eliminar el objetivo de "500 variables":**
   La calidad de un dataset no se mide por su ancho, sino por su señal. Un dataset de 40 columnas con señal táctica real es infinitamente superior a esta matriz inflada de 540 columnas.
3. **Rehacer el PCA con datos limpios:**
   Una vez que tengan un dataset racional (<80 columnas), apliquen PCA. Verán que la varianza se distribuye de manera mucho más natural y que los "Loadings" de cada componente sí tendrán verdadero sentido de negocio, sin estar inflados por multicolinealidad extrema.
4. **Validar la imputación de nulos:**
   Revisen qué porcentaje del dataset final son valores imputados. Si una variable tiene un 40% de nulos imputados por la mediana, su varianza es falsa y engañará al PCA. Elimínenla.

Espero ver la "Versión 4" de este dataset mañana por la mañana con estas correcciones implementadas. Dejen de lado la "fuerza bruta" y comiencen a aplicar ciencia de datos real.
