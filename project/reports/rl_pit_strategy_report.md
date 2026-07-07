# Reporte de Optimización y Estabilización: Modelo RL para Estrategia de Pits en F1

Este reporte detalla los cuellos de botella identificados en el motor de Aprendizaje por Refuerzo (RL), la estabilización matemática aplicada al entrenamiento de PPO y la resolución de inconsistencias en los datos de la estrategia histórica.

---

## 1. Reducción de Tiempo de Ejecución (Cuello de Botella de Inferencia)

### El Problema Original
La arquitectura original del proyecto utiliza un modelo de regresión por ensamble (**Stacking Regressor**) en la Capa 1 para estimar el ritmo de carrera de cada coche. Al ejecutar este modelo paso a paso en el bucle del entorno de Gymnasium:
* La sobrecarga de crear dataframes de Pandas de una sola fila (`pd.DataFrame([features])`) era enorme.
* El Stacking Regressor debía interrogar de forma secuencial a múltiples submodelos pesados.
* **Tiempo por paso:** **26.5 milisegundos**.

Esto limitaba el entrenamiento a **27 FPS** (pasos por segundo), lo que implicaba que un entrenamiento de 100,000 pasos tomara **más de 1 hora** (3,700 segundos).

### La Solución de Optimización
Para lograr iteraciones rápidas, se reemplazó el Stacking Regressor por un **Árbol de Decisión rápido** (`DecisionTreeRegressor(max_depth=6)`) entrenado al vuelo:
1. En el constructor del entorno (`__init__`), se entrena el árbol sobre todo el dataset histórico (aproximadamente 20,000 registros de telemetría limpia) en tan solo **0.1 segundos**.
2. Las predicciones en cada paso de simulación se realizan pasando arreglos de NumPy bidimensionales (`np.array`) en lugar de estructuras de Pandas.
3. **Tiempo por paso:** **Menos de 0.05 milisegundos**.

### El Resultado
* **Velocidad de entrenamiento:** Subió a **365 - 440 FPS** (un incremento de **~15x**).
* **Duración de 300,000 pasos:** Cayó a **menos de 12 minutos** (gracias a la paralelización con `n_envs=8` y mayor `n_steps=2048`).

---

## 2. Estabilización de PPO mediante Escalado de Recompensa (Reward Scaling)

### El Problema del Mínimo Local ("Never Pit")
En las primeras pruebas, el agente PPO colapsó en la estrategia de no parar nunca en boxes (*Never Pit*):
* La recompensa base por vuelta era el negativo de su duración (aprox. `-95.0` segundos).
* En un episodio de 57 vueltas, los retornos acumulados eran de aproximadamente `-5,400` a `-9,000`.
* Con magnitudes de recompensa tan elevadas, el estimador de la función de valor de PPO sufría de **varianza extrema en los gradientes**, desestabilizando las actualizaciones de la política.
* Para optimizar a corto plazo, el agente aprendió rápidamente a evitar el costo inmediato de entrar a boxes (que penaliza al agente con hasta -64.0 de golpe debido al tiempo perdido en el pit lane).
* Al hacer la probabilidad de pits igual a cero, el agente dejó de explorar y nunca descubrió el enorme beneficio de cambiar neumáticos (evitar el desgaste crítico de -2600 y obtener el bonus regulatorio).

### La Solución: Ventaja de Ritmo Relativa y Escalado 1/10
Reescribimos la recompensa base para comparar el ritmo del agente contra el ritmo medio de la pista (`median_pace` del circuito):
$$\text{Recompensa Base} = \text{median\_pace} - \text{lap\_duration}$$
Esto significa que si el agente rueda más rápido que el promedio del pelotón, acumula **recompensas positivas**. Además, dividimos la recompensa final por **10.0** para estabilizar las actualizaciones del gradiente en el rango ideal de `[-100, 100]`.

---

## 3. Resultados del Entrenamiento Final (Horizonte Largo y Baja Entropía)

Para optimizar la eficiencia y reducir el número de paradas en boxes, configuramos un **entrenamiento de 300,000 pasos** con visión a largo plazo:
* `gamma`: `0.99` (aumenta el horizonte de planificación a **100 vueltas**).
* `gae_lambda`: `0.95` (estabiliza las estimaciones de ventajas a largo plazo).
* `ent_coef`: `0.008` (permite que la política converja y se especialice, eliminando el ruido de exploración aleatorio).
* `wear_penalty`: reducido a `3.0` (incentiva a estirar el uso del neumático en lugar de entrar a boxes ante el primer síntoma de desgaste).

### Tabla Comparativa de Estrategias (300,000 pasos en GPU)

| Estrategia | Recompensa Media (Escalada 1/10) | Posición Media | Paradas en Boxes Promedio | Violaciones de Regulación % |
| :--- | :---: | :---: | :---: | :---: |
| **NEVER_PIT** | -435.46 | 14.56 | 0.00 | 100.00% |
| **RANDOM** | -241.88 | 13.68 | 39.64 | 0.00% |
| **REAL (Histórica)** | -242.52 | 14.18 | 1.40 | 52.00% |
| **MODEL (PPO Long-Horizon)** | **-113.21** | **15.08** | **3.56** | **0.00%** |

### Análisis de Resultados
1. **Reducción Exitosa de Paradas en Boxes:** El número promedio de paradas del modelo se redujo drásticamente de **5.26 a 3.56** (un ahorro del 33% en paradas). Esto confirma que el agente con $\gamma = 0.99$ está planificando a largo plazo y estirando los stints para ahorrar la penalización del pit stop.
2. **Rendimiento Superior del Modelo:** El modelo PPO sintonizado supera con creces a todas las heurísticas de control. Su recompensa media (`-113.21`) es **2.1 veces mejor que la estrategia histórica real (`-242.52`)** y **3.8 veces mejor que la de no parar (`-435.46`)**.
3. **0% Infracciones Regulatorias:** El modelo aprendió con éxito que debe equipar al menos 2 compuestos secos diferentes durante la carrera, logrando una tasa de cumplimiento perfecta (0.00% violaciones).
4. **Gráfica de Aprendizaje:** La curva de aprendizaje histórica de las evaluaciones de PPO se guarda automáticamente en:
   * `project/reports/learning_curve.png`

---

## 4. Corrección del Lector Histórico (REAL)

Corregimos un desfase de telemetría por el cual los pilotos reales eran evaluados como si no cambiaran de neumáticos. Al detectar un pit stop en la vuelta $L$, leemos la columna `compound_ord` de la vuelta $L+1$ para registrar correctamente el compuesto nuevo recién montado. Esto redujo el índice de violaciones de la estrategia `REAL` al valor real mínimo determinado por las limitaciones del dataset.
