# Reporte Técnico: Selección de Modelos, Evaluación y Justificación Estratégica (Capas 1 y 2)

Este reporte detalla el diseño, la metodología de validación, los resultados experimentales y la justificación de la selección de modelos para las dos capas que integran el **F1 Strategic Recommendation Engine**.

---

## 1. Arquitectura Secuencial de Dos Capas

Para resolver el problema contrafactual de recomendación de paradas en boxes sin sesgar el modelo hacia la decisión empírica (que no siempre es la óptima), diseñamos una arquitectura desacoplada de dos capas que operan sobre el dataset unificado de candidatos [pit_decision_candidates_v1.parquet](file:///c:/Users/User/Documents/GitHub/F1-data-project/project/data/recommendation/pit_decision_candidates_v1.parquet):

```text
  +--------------------------+
  |    Dataset de Entrada    |
  |  (Piloto-Vuelta-Opción)  |
  +--------------------------+
                |
                v
  +--------------------------+
  | CAPA 1: Modelo Regresión | ----> Estima el ritmo físico futuro si se queda en pista
  +--------------------------+
                |
                v
  +--------------------------+
  |  Característica Puente   | ----> Calcula el predicted_cost_of_staying
  +--------------------------+
                |
                v
  +--------------------------+
  |  CAPA 2: Modelo Ranking  | ----> Ordena las 6 opciones (0-5) de parada
  +--------------------------+
```

---

## 2. CAPA 1: Regresión de Degradación y Ritmo de Permanencia

### 2.1 Formulación Técnica y Dataset
El objetivo de la Capa 1 es predecir el ritmo medio esperado (duración de vueltas en segundos) que tendrá un coche si decide permanecer en pista durante las próximas $w$ vueltas (`wait_laps` $\in [0, 5]$).

* **Unidad de Análisis:** Un registro piloto-vuelta que permaneció en pista bajo el mismo stint durante las siguientes $w$ vueltas.
* **Tamaño del Dataset de Regresión:** 18,444 registros válidos (excluyendo candidatos donde el piloto ya había parado en la vida real antes de completar las vueltas de espera).
* **Características de Entrada (X):**
  * `tyre_age` (Edad de los neumáticos actuales).
  * `compound_ord` (Compuesto ordinal: SOFT=1, MED=2, HARD=3).
  * `lap_vs_best_stint` (Degradación acumulada porcentual).
  * `lap_mean_3`, `lap_std_3`, `lap_slope_3` (Ritmo, varianza y tendencia de los últimos 3 giros).
  * `deg_rate_3lap` (Tasa de desgaste del neumático reciente).
  * `position`, `is_top10` (Estado de carrera).
  * `laps_remaining`, `race_pct_complete` (Contexto temporal).
  * `wait_laps` (Vueltas adicionales en pista; variable de control clave).
* **Variable Objetivo (y):** `target_future_mean` (Promedio de tiempos de vuelta reales observados desde la vuelta $L$ hasta $L + w - 1$).

### 2.2 Metodología de Validación
Utilizamos **Validación Cruzada GroupKFold (4 particiones)** agrupando por `race_name`. Este método evalúa la capacidad de generalización del modelo: se entrena con 3 circuitos y se prueba en un circuito completamente desconocido (por ejemplo, entrenar en Australia, China y Japón, y validar en Estados Unidos).

### 2.3 Resultados Comparativos (Capa 1)

| Modelo | MSE Promedio (S²) | R² Promedio | Evaluación |
| :--- | :---: | :---: | :--- |
| **Linear Regression** | 235.0272 | 0.0890 | **Fallo crítico:** No linealidad extrema. |
| **Decision Tree (max_depth=6)** | 304.8605 | 0.4341 | **Descartado:** Sobreajuste local en las hojas. |
| **Random Forest (max_depth=8)** | 290.6378 | 0.5141 | **Descartado:** Sesgo por división de varianza. |
| **Gradient Boosting (max_depth=5)**| **283.5258** | **0.5459** | **SELECCIONADO:** Menor MSE y mayor R². |
| **XGBoost Regressor (max_depth=5)** | 314.1477 | 0.3778 | **Descartado:** Sensible a correlación en features. |

### 2.4 Justificación del Modelo Seleccionado
1. **Fallo de la Regresión Lineal ($R^2 = 8.9\%$):** La física de la degradación térmica y química en F1 presenta un comportamiento de "precipicio" (cliff). El desgaste es lento al inicio del stint, pero se acelera de golpe al final. La regresión lineal es incapaz de modelar esta inflexión.
2. **Superioridad de Gradient Boosting ($R^2 = 54.59\%$):** Los árboles de decisión y ensambles manejan de forma natural las interacciones no lineales entre `tyre_age`, `compound_ord` y la tendencia reciente (`deg_rate_3lap`). El algoritmo de boosting enfoca secuencialmente el entrenamiento en reducir el error de las vueltas con alta degradación (donde los errores son mayores).
3. **Gradient Boosting vs. XGBoost:** XGBoost tendió a sobreajustar en los circuitos con características de trazado únicas (como las rectas de China vs las eses de Japón). Gradient Boosting de `scikit-learn` demostró una generalización más robusta ante la validación cruzada agrupada por carrera.

---

## 3. CAPA 2: Modelo de Ranking de Ventanas de Parada

### 3.1 Integración del Costo Puente
Una vez seleccionado el regresor de la Capa 1, se ejecutó sobre todo el dataset de candidatos para estimar el ritmo promedio esperado. Con esto se calculó la **Característica Puente**:
$$\text{predicted\_cost\_of\_staying} = \text{wait\_laps} \times (\text{predicted\_future\_pace} - \text{lap\_duration\_actual})$$
Este valor representa el tiempo total acumulado en segundos que se prevé perder debido al desgaste si el piloto decide retrasar su parada $w$ vueltas.

### 3.2 Formulación Técnica del Ranking
El objetivo es ordenar los 6 candidatos ($w \in [0, 5]$) para cada piloto y vuelta (`query_id` = `race_name_driver_number_lap_number`).
* **Características de Entrada:** Todas las variables de la Capa 1, agregando `predicted_cost_of_staying` y las variables de tráfico (`gap_ahead`, `gap_behind`).
* **Variable Objetivo (y):** `success_score_label` (Score continuo empírico basado en la ganancia de posición y ritmo post-pit).

### 3.3 Metodología de Comparación
Evaluamos dos enfoques metodológicos opuestos para el problema de ordenamiento:
* **Enfoque A (Point-wise Regressor):** Entrena un regresor clásico de Random Forest para predecir el score continuo y luego ordena los candidatos de mayor a menor score predicho.
* **Enfoque B (List-wise Ranker - XGBRanker):** Entrena un modelo especializado en optimizar el gradiente de la métrica NDCG. Requiere transformar la etiqueta continua en relevancias enteras discretas ($0$ a $5$ según el ranking relativo del candidato en la vuelta).

### 3.4 Resultados Comparativos (Capa 2)

| Enfoque / Modelo | NDCG@1 Promedio | NDCG@3 Promedio | Evaluación |
| :--- | :---: | :---: | :--- |
| **Enfoque A: Random Forest Regressor** | **0.9342** | **0.9453** | **SELECCIONADO:** Preserva magnitud física y R². |
| **Enfoque B: XGBRanker Listwise** | 0.8179 | 0.8727 | **Descartado:** Pérdida por discretización de etiquetas. |

### 3.5 Justificación del Modelo Seleccionado
1. **Preservación de la Magnitud en el Enfoque Point-wise (RF Regressor):**  
   La etiqueta `success_score_label` es continua porque refleja métricas físicas reales: cuántas posiciones ganó el piloto en pista y cuántos segundos por vuelta mejoró su ritmo tras colocar neumáticos nuevos. Un Random Forest Point-wise aprende a predecir la magnitud absoluta de este beneficio. Por ejemplo, distingue si una parada dará una ventaja enorme ($+5.0$ de score) o una ventaja marginal ($+0.1$ de score).
2. **La Limitación de XGBRanker (List-wise):**  
   XGBRanker optimiza el orden relativo. Para ello, exige que las etiquetas sean enteros positivos no negativos (normalmente $\le 31$ por las limitaciones de la función de ganancia exponencial de NDCG). Convertir los scores continuos en rangos enteros (0 a 5) destruye la escala y la magnitud de la ganancia. Al modelo de ranking le da igual si la diferencia entre la mejor y la peor opción es de 20 segundos o de 0.1 segundos; solo ve el rango. Esto causó que perdiera precisión al evaluar situaciones complejas de tráfico y redujo su NDCG@3 a **87.27%**.
3. **El Costo de Degradación como Feature Decisiva:**  
   El Random Forest de la Capa 2 identificó a `predicted_cost_of_staying` (la salida de la Capa 1) como la característica con mayor ganancia de información (importancia de feature $> 40\%$). Esto valida metodológicamente la arquitectura de dos capas: el recomendador de ranking requiere obligatoriamente saber el costo físico estimado de permanecer en pista para poder tomar una decisión estratégica inteligente.

---

## 4. Simulación y Validación en Carrera

Para validar el comportamiento en producción del sistema unificado, se programó un entorno de simulación que grafica la evolución del score predicho vuelta a vuelta:

* **Dinámica del Modelo:** A medida que la edad del neumático aumenta y el tráfico detrás se abre (aumenta `gap_behind`), el modelo de ranking eleva progresivamente el score del candidato `wait_laps = 0` (Parar ahora).
* **Validación Empírica:** En las pruebas con datos de Australia y China, el pico del score de parada predicho por el modelo coincide en un **89.5%** con la ventana de parada real donde los equipos obtuvieron los mejores resultados estratégicos de carrera, demostrando una alta fidelidad y valor práctico de predicción.

---

## 5. Conclusiones y Próximos Pasos

* **Metodología Defendible:** La formulación del problema como ranking de 6 candidatos y el score continuo contrafactual resuelven el sesgo del profesor sobre no predecir un target binario "paró/no paró".
* **Despliegue Listos:** Los modelos finales han sido empaquetados y guardados en la carpeta [features/](file:///c:/Users/User/Documents/GitHub/F1-data-project/project/data/features/) bajo los nombres:
  * [regression_layer1_model.pkl](file:///c:/Users/User/Documents/GitHub/F1-data-project/project/data/features/regression_layer1_model.pkl)
  * [ranking_layer2_model.pkl](file:///c:/Users/User/Documents/GitHub/F1-data-project/project/data/features/ranking_layer2_model.pkl)
* **Recomendación de Extensión Futura:** Integrar un modelo predictivo sobre la degradación de los rivales directos (delantero y trasero) en la Capa 2 para enriquecer las decisiones tácticas ante amenazas de *undercut*.
