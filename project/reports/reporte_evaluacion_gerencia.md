# REPORTE DE EVALUACIÓN DE MODELOS DE CLUSTERING
**De:** Gerencia del Proyecto de Datos F1
**Para:** Equipo de Data Science
**Estado:** CRÍTICO / SE REQUIEREN ACCIONES INMEDIATAS

He revisado los notebooks de *Clustering* (K-Means, Hierarchical y DBSCAN) que el equipo entregó. No voy a usar adornos: el trabajo presentado está lleno de inconsistencias, sesgos de confirmación y falta de rigor analítico. Tienen métricas frente a ustedes y están tomando decisiones que las contradicen directamente para forzar una narrativa de negocio.

A continuación detallo el análisis crudo de por qué estos modelos, en su estado actual, no sirven para producción y qué deben hacer para arreglarlos.

---

## 1. K-Means: Selección de Parámetros Injustificable

Ustedes implementaron un K-Means y realizaron un "barrido" (sweep) de hiperparámetros. Las métricas generadas por su propio código son claras:

| k | Inertia (WCSS) | Silhouette Score | Calinski-Harabasz | Davies-Bouldin |
|---|---|---|---|---|
| 2 | 143810.4 | 0.3721 | 341.04 | 1.3103 |
| **3** | **117558.9** | **0.4113** | **279.73** | **1.1696** |
| 4 | 105847.6 | 0.3910 | 230.36 | 1.4782 |

**¿Por qué demonios concluyen que k=4 es el "número óptimo"?**
Tienen un Silhouette más alto en `k=3` (0.41 vs 0.39), el Calinski-Harabasz es muy superior (279 vs 230), y el error Davies-Bouldin es el más bajo y mejor en `k=3` (1.16 vs 1.47). Ignoraron sus propias métricas para justificar `k=4` basándose en una evaluación visual y subjetiva del "codo de inercia", la cual es la métrica más débil de todas. 

Esto es inaceptable. Están forzando un cuarto clúster que no existe matemáticamente porque seguramente encajaba en su idea preconcebida de las tácticas. 

### Análisis de Fallos del K-Means (El "Cluster Basura")
El análisis de fallos revela que tienen 25 puntos con un Silhouette negativo (es decir, puntos mal asignados que están más cerca de otros clústeres que del suyo). Lo peor: **25 de esos errores están en el Clúster 3**, y **24 de ellos corresponden a eventos de `On_Track_Overtake`**. 
- **Conclusión directa:** Su Clúster 3 es basura. Es un grupo forzado que no logra encapsular de manera limpia los adelantamientos en pista. Esto ocurre, irónicamente, porque forzaron `k=4`.

---

## 2. Hierarchical Clustering (Ward): Rendimiento Mediocre

El modelo jerárquico reporta un Silhouette de **0.2530** y un Calinski-Harabasz de **219.40**.
No sé quién redactó el apartado de conclusiones diciendo que *"El algoritmo es capaz de separar los asaltos de DRS... en curvas"*, pero con un Silhouette de 0.25, sus clústeres están altamente superpuestos y apenas logran separarse más allá del ruido estadístico. 

Un Silhouette de 0.25 en este contexto espacial de 15 componentes principales indica que las fronteras que Ward está dibujando son difusas. Decir que han encontrado el "ADN de los ataques" basándose en esto es vender humo. El modelo jerárquico es inferior a K-Means en este dataset y su uso no justifica la toma de decisiones estratégicas. Descarten este modelo o úsenlo exclusivamente como herramienta de exploración secundaria, no como un pipeline final.

---

## 3. DBSCAN: El Modelo "Caja Negra"

Abrí el notebook de DBSCAN esperando ver la proporción de ruido (porcentaje de outliers clasificados como `-1`), la métrica Silhouette (sin ruido) y la gráfica de k-distancias para justificar `eps` y `min_samples`.
¿Qué encontré? Nada riguroso reportado al final. Solo un mensaje de "Resultados espaciales listos para exportar". 
DBSCAN en un espacio de 15 dimensiones (PCA) sufre terriblemente de la maldición de la dimensionalidad. Sin un reporte explícito del porcentaje de datos descartados como ruido, asumo que el modelo o clasificó la mitad del dataset como ruido, o agrupó todo en un solo clúster. 
- **Conclusión:** Si no pueden mostrar las métricas de validación frente a mi cara, asumo que el modelo no sirve.

---

## RECOMENDACIONES CRUDAS Y PASOS A SEGUIR

Tienen que rehacer la validación y limpieza de conclusiones. Mis instrucciones son claras:

1. **Revertir a K=3 en K-Means:** 
   El K-Means es su modelo más rescatable (0.41 en Silhouette), pero deben usar `k=3`. Dejen de inventarse clústeres. Ejecuten el notebook nuevamente con `k=3`, verifiquen si el problema de puntos mal asignados (negativos) desaparece. Perfilar esos 3 arquetipos tácticos reales.
2. **Justificar la retención de K=4 matemáticamente (Si insisten):**
   Si están completamente seguros de que el negocio *necesita* 4 clústeres, deben demostrar por qué `k=4` aporta valor de negocio, admitiendo abiertamente en el reporte que matemáticamente es un modelo subóptimo y asumiendo la degradación del Clúster 3. Nada de engaños.
3. **Descartar Hierarchical Clustering de la entrega final:**
   No lo utilicen para generar insights. Las métricas (0.25) no lo respaldan frente a K-Means.
4. **Documentar DBSCAN o Eliminarlo:**
   Si DBSCAN no logró separar los clústeres en 15 dimensiones (lo cual es muy probable), documenten formalmente el fracaso. Digan: *"DBSCAN falló en converger de manera útil debido a la dispersión en 15 dimensiones con un ratio de ruido de X%"*. Saber qué no funciona también es útil, pero no entreguen notebooks mudos.
5. **Enfocarse en la métrica, no en la narrativa:**
   El equipo está intentando que los datos digan lo que ustedes quieren escuchar. Que el modelo hable primero, y ustedes pónganle nombre a los grupos después.

Tienen el día de hoy para corregir estos enfoques. No quiero volver a leer una justificación basada en "el codo de inercia" cuando las métricas duras dicen lo contrario.
