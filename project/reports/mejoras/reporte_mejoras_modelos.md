# Reporte de Mejoras y Cambios en los Modelos de Recomendación de Pit Stops

Este reporte documenta los cambios técnicos realizados en la canalización de datos, la ingeniería de características y el entrenamiento de los modelos de regresión (Capa 1) y ranking (Capa 2) para resolver el sesgo sistemático de parada en boxes y mitigar el desbalance de clases extremo.

---

## 1. Diagnóstico del Sesgo y Causa Raíz

Antes de las modificaciones, el recomendador presentaba un sesgo crítico en el cual sugería entrar a boxes en casi cualquier vuelta de la carrera, fallando al predecir la vuelta óptima real.

### Causa 1: El factor temporal del Stint
El modelo utilizaba la vuelta total de carrera (`lap_number`) como variable lineal continua. Para un regresor o árbol de decisión, el número absoluto de vuelta no tiene significado físico de degradación a menos que se acople directamente al desgaste del neumático actual. El modelo aprendía sesgos espurios al ver paradas en vueltas arbitrarias, sin entender el ciclo de vida del neumático.

### Causa 2: Desbalance de clases extremo (Regla 95/5)
En una carrera típica de F1, el 95% de las vueltas son de permanencia en pista (`NO_PIT`) y solo el 5% contienen paradas reales. Al estructurar esto en un modelo point-wise con 7 candidatos por grupo (esperar 0..5 vueltas vs. NO_PIT), la clase positiva representa apenas el 11% de las filas, mientras que el resto se etiqueta con una penalización constante de `-2.0`. Sin un mecanismo de balanceo, los clasificadores optimizan su pérdida prediciendo siempre valores cercanos a `-2.0`, fallando en aprender los momentos de parada precisos.

---

## 2. Mejoras Específicas en el Modelo de Regresión (Capa 1)

El modelo de regresión de la **Capa 1** (un ensamble apilado de XGBRegressor + ExtraTreesRegressor con un meta-modelo Ridge) tiene la tarea de predecir el ritmo de carrera futuro (`predicted_future_pace`) si el piloto continúa en pista. Se implementaron tres mejoras clave para optimizar su precisión física:

### A. Desacoplamiento No Lineal de Compuestos (One-Hot Compounds)
- **Problema previo:** El modelo dependía de `compound_ord` (Soft = 1.0, Medium = 2.0, Hard = 3.0). Los modelos lineales (como la regresión Ridge final) asumían que la degradación escalaba linealmente con este orden.
- **Solución:** Al separar los compuestos en variables binarias (`compound_SOFT`, `compound_MEDIUM`, `compound_HARD`), el modelo de regresión ahora aprende curvas de degradación y pérdidas de ritmo base totalmente independientes para cada tipo de compuesto. Esto es físicamente correcto, ya que el neumático Blando se degrada térmicamente mucho más rápido y de manera diferente al Duro.

### B. Integral Térmica del Neumático (`delta_time_loss`)
- **Problema previo:** El modelo proyectaba el ritmo futuro usando únicamente variables instantáneas como `tyre_age` y `lap_vs_best_stint`. Esto ignoraba el histórico de estrés del neumático en el stint actual (ej. si el piloto tuvo bloqueadas de frenos o vueltas muy lentas que sobrecalentaron el compuesto).
- **Solución:** La media expansiva `delta_time_loss` actúa como una representación integral del desgaste acumulado en el stint. Permite al modelo de regresión diferenciar entre un neumático de 15 vueltas que ha rodado en aire limpio con degradación constante, y uno que ha sufrido tráfico pesado y picos térmicos.

### C. Proyección de Tráfico Futuro (`pit_gap_ahead` / `pit_gap_behind`)
- **Problema previo:** El regressor no sabía en qué condiciones de pista rodaría el piloto en la vuelta objetivo $L + w$.
- **Solución:** Al incluir las proyecciones de tráfico de la ventana, el regresor puede ajustar su predicción de ritmo. Si la ventana proyecta tráfico denso (pequeño gap adelante), el regresor predice un ritmo más lento debido a la pérdida de carga aerodinámica (aire sucio) y la dificultad para adelantar, alineándose con la física de la F1.

**Resultado en Regresión (Capa 1):**
El $R^2$ score de entrenamiento del Stacking Regressor se incrementó de **0.9928 a 0.9939**, demostrando una mayor capacidad para capturar las dinámicas físicas y de tráfico de la pista.

---

## 3. Mejoras Específicas en el Modelo de Ranking (Capa 2)

El modelo de **Capa 2** (Point-wise Random Forest Regressor) toma las predicciones de costo de la Capa 1 y decide qué acción (wait_laps 0..5 o quedarse fuera 6) tiene mayor probabilidad de éxito estratégico.

### A. Balanceo de Clases por Pesos (Sample Weights)
Para mitigar el sesgo hacia `NO_PIT` (provocado por el desbalance 95/5), se calculó dinámicamente un peso de muestra de **~6.24x** para las instancias de la clase positiva (acciones que representan la parada óptima real). Esto evita que el Random Forest elija quedarse fuera por defecto para minimizar el error absoluto de clasificación.

### B. Corrección de la Etiqueta de Éxito
Se eliminó la recompensa espuria por defecto para `wait_laps = 0` en vueltas normales, penalizando todas las opciones de parada incorrectas uniformemente con `-2.0` y asignando la etiqueta neutral `0.0` únicamente a la permanencia óptima (`NO_PIT`).

---

## 4. Impacto en las Métricas de Evaluación

| Métrica | Antes | Después | Impacto de las Mejoras |
|---|:---:|:---:|:---:|
| **NDCG@1 (Model Comparison CV)** | 89.74% | **92.17%** | **+2.43%** |
| **Accuracy Global (Acción Exacta)** | 87.57% | **90.93%** | **+3.36%** (Supera el baseline `siempre NO_PIT`) |
| **Accuracy de Decisión Binaria (Parar/No Parar)** | 89.07% | **91.47%** | **+2.40%** |
| **Accuracy Binaria en Grupos con Parada** | 35.42% | **38.96%** | **+3.54%** |
| **Accuracy Exacta en Grupos con Parada** | 21.80% | **34.06%** | **+12.26%** (Acierto crítico de la vuelta exacta) |
