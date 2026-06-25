# Week 12: Graph Analytics and Centrality Report

Este reporte presenta la formalización, modelado y análisis estructural de los adelantamientos en pista (*On-Track Overtakes*) para la temporada 2026 de Fórmula 1 utilizando teoría de grafos y métricas de red.

---

## 1. Definición Formal del Grafo y Justificación

Para modelar la competitividad y la dominancia en pista de manera pura, el grafo se define formalmente de la siguiente manera:

*   **Nodos ($V$):** Pilotos activos en la carrera o temporada, identificados por sus siglas oficiales de tres letras (ej: `VER`, `HAM`, `NOR`). Mapeados desde los archivos `drivers.csv`.
*   **Aristas ($E$):** Enlaces dirigidos que representan adelantamientos exitosos en pista. Los enlaces van desde el piloto adelantado (defensor) hacia el piloto que realiza el adelantamiento (atacante):
    $$B \rightarrow A \quad (\text{Piloto Adelantado } B \text{ apunta al Piloto que Adelanta } A)$$
*   **Pesos ($W$):** La cantidad acumulada de adelantamientos exitosos en pista entre un par específico de pilotos.
*   **Direccionalidad:** Dirigido.

### Justificación de las Decisiones de Diseño

1.  **Enfoque en Adelantamientos en Pista (*On-Track Overtakes*):** Al filtrar los eventos y excluir paradas en boxes (*undercut* / *overcut*) o retiros por fallas mecánicas, el grafo mide exclusivamente el rendimiento de combate rueda a rueda y la dominancia de pilotaje.
2.  **Dirección del Flujo ($B \rightarrow A$):** En la teoría de redes, el PageRank distribuye el "prestigio" o la "dominancia" de un nodo a través de sus enlaces salientes. Al hacer que las aristas apunten del piloto superado al que supera, el piloto que realiza el adelantamiento recibe una transferencia de dominancia. Si un piloto adelanta a un rival altamente calificado (que a su vez tiene un alto PageRank porque adelanta a otros y rara vez es superado), recibirá una transferencia de rango mucho mayor. Esto crea un ranking de combatividad cualitativo muy superior al conteo de adelantamientos simple.
3.  **Inclusión de todos los pilotos como nodos:** Todos los pilotos inscritos en la carrera se inicializan como nodos en el grafo, incluso si terminan con grado 0 (sin adelantamientos hechos ni sufridos). Esto permite identificar pilotos que corrieron carreras solitarias en aire limpio o aislados en el fondo de la parrilla.

---

## 2. Reporte de Métricas: GP de Australia 2026

Al ejecutar el script de análisis sobre la carrera de Australia 2026, se obtuvieron las siguientes métricas de centralidad:

### Tabla de Métricas de Red (Australia 2026)

| Piloto | Adelantamientos Hechos (Ofensiva) | Veces Superado (Defensiva) | PageRank (Dominancia) | Centralidad de Intermediación (*Betweenness*) |
| :--- | :---: | :---: | :---: | :---: |
| **NOR** | 11 | 8 | 0.084273 | 0.030992 |
| **BEA** | 6 | 7 | 0.071950 | 0.063452 |
| **SAI** | 6 | 7 | 0.069561 | 0.187579 |
| **VER** | 9 | 3 | 0.067503 | 0.071508 |
| **ALB** | 8 | 2 | 0.066953 | 0.055159 |
| **RUS** | 3 | 2 | 0.066748 | 0.002381 |
| **LAW** | 5 | 5 | 0.065385 | 0.071825 |
| **COL** | 5 | 7 | 0.055738 | 0.137897 |
| **HAM** | 3 | 1 | 0.051414 | 0.000000 |
| **LIN** | 3 | 9 | 0.044745 | 0.009048 |

### Análisis de Componentes Conexas (Australia 2026)

El análisis reveló **5 componentes conexas independientes** (pelotones aislados):

*   **Grupo 1 (16 pilotos):** `OCO, STR, BEA, ALB, BOR, ANT, PER, SAI, NOR, BOT, LAW, VER, LIN, GAS, COL, ALO`. Este es el pelotón principal o "núcleo de batalla". Los pilotos de este grupo interactuaron directa o indirectamente en pista.
*   **Grupo 2:** `HAD` (Completamente aislado. Corrió en solitario sin luchas).
*   **Grupo 3:** `RUS, LEC, HAM` (Un trío que luchó de forma aislada. Russell, Leclerc y Hamilton tuvieron duelos directos, pero no interactuaron mediante adelantamientos con el pelotón principal).
*   **Grupo 4:** `HUL` (Aislado).
*   **Grupo 5:** `PIA` (Aislado).

---

## 3. Reporte de Métricas: Grafo Global de la Temporada

El análisis consolidado de todas las carreras de la temporada (Australia, China, Japón y Estados Unidos) genera un grafo mucho más denso y conectado:

*   **Nodos totales (Pilotos):** 22
*   **Aristas (Duelos únicos en pista):** 341

### Tabla de Métricas de Red Globales (Top 10)

| Piloto | Adelantamientos Hechos (Ofensiva) | Veces Superado (Defensiva) | PageRank (Dominancia) | Centralidad de Intermediación (*Betweenness*) |
| :--- | :---: | :---: | :---: | :---: |
| **VER** | 46 | 30 | 0.073675 | 0.040446 |
| **OCO** | 45 | 40 | 0.073232 | 0.014492 |
| **BOR** | 37 | 34 | 0.061600 | 0.027319 |
| **LEC** | 33 | 35 | 0.055386 | 0.017494 |
| **PER** | 33 | 36 | 0.055219 | 0.034140 |
| **ANT** | 30 | 24 | 0.051754 | 0.035845 |
| **LIN** | 28 | 44 | 0.051256 | 0.014707 |
| **ALB** | 31 | 27 | 0.051175 | 0.053037 |
| **NOR** | 30 | 24 | 0.050327 | 0.024489 |
| **BEA** | 27 | 27 | 0.045920 | 0.031709 |

---

## 4. Sección Comparativa: PageRank vs. Baselines Externos (Posición de Carrera) y Modelos

Para validar la utilidad del ranking estructural del grafo de adelantamientos, es fundamental contrastar los resultados de centralidad frente a **baselines externos estructurados** y no solo contra métricas internas del grafo (como el volumen de adelantamientos/popularidad).

### A. Comparación Estructurada: PageRank vs. Posición Real de Llegada (GP de Australia 2026)

El siguiente análisis contrasta el orden de dominancia por PageRank frente a la posición de llegada real en la carrera de Australia 2026:

| Piloto | PageRank | Rank PageRank | Posición Llegada (Real) | Diferencia (Posición - PageRank) | Interpretación y Dinámica Deportiva |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **NOR** | 0.084273 | 1 | 6 | +5 | **Combate en Tráfico:** Norris remontó con 11 adelantamientos y solo 8 sufridos, obteniendo la mayor dominancia en pista a pesar de no ganar la carrera. |
| **BEA** | 0.071950 | 2 | 7 | +5 | **Zona Media Activa:** Mucha combatividad con pilotos competitivos en el pelotón medio alto. |
| **SAI** | 0.069561 | 3 | 15 | +12 | **Alta Actividad, Baja Eficiencia:** Sainz luchó constantemente en pista pero cayó al 15º final, demostrando que PageRank mide batallas y no el resultado final. |
| **VER** | 0.067503 | 4 | 5 | +1 | Relación lineal estrecha entre su ritmo de carrera y su actividad de adelantamientos. |
| **ALB** | 0.066953 | 5 | 13 | +8 | Batalló agresivamente en el tráfico, aunque su coche no le permitió terminar en los puntos. |
| **RUS** | 0.066748 | 6 | 1 | -5 | **Ganador en Aire Limpio:** Russell ganó la carrera liderando desde adelante. Al no estar en tráfico, tuvo muy pocos duelos y su PageRank es bajo en comparación. |
| **LAW** | 0.065385 | 7 | 12 | +5 | Batalló bastante en tráfico en el fondo de los puntos. |
| **COL** | 0.055738 | 8 | 14 | +6 | Muy activo en la zona media defensiva, actuando de tapón y acumulando interacciones. |
| **HAM** | 0.051414 | 9 | 2 | -7 | **Podio Aislado:** Poca interacción de adelantamientos con el pelotón de en medio. |
| **LIN** | 0.044745 | 10 | 9 | -1 | Mantiene una buena correlación entre su ritmo real y su combatividad. |
| **OCO** | 0.042931 | 11 | 11 | 0 | Relación directa entre adelantamientos hechos/sufridos y su posición final. |
| **BOT** | 0.041670 | 12 | 19 | +7 | Tuvo algunas luchas en el fondo antes de retirarse. |
| **ANT** | 0.039800 | 13 | 4 | -9 | Corrió usualmente al frente, por lo que su PageRank es menor que su consistencia. |
| **GAS** | 0.039322 | 14 | 10 | -4 | Se mantuvo en el fondo de los puntos con interacciones limitadas. |
| **ALO** | 0.038619 | 15 | 18 | +3 | Luchó brevemente en la zona media antes de su retiro. |
| **BOR** | 0.037442 | 16 | 8 | -8 | Poca combatividad ofensiva a pesar de rescatar un buen 8º puesto. |
| **LEC** | 0.036080 | 17 | 3 | -14 | **Aislamiento en el Frente:** Leclerc terminó en el podio (3º), pero al estar aislado en el Grupo 3 casi no adelantó ni fue adelantado, hundiéndose en PageRank. |

### B. Comparación Consolidad Temporada: PageRank Global vs. Posición Promedio en Carrera

Al consolidar los datos de toda la temporada (Australia, China, Japón y Estados Unidos), el PageRank acumulado se contrasta contra el rango promedio de finalización en carrera de cada piloto:

| Piloto | PageRank Global | Rank PageRank | Rango Promedio Carrera | Rank Promedio Carrera | Diferencia de Rank | Dinámica de la Temporada |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **VER** | 0.073675 | 1 | 6.67 | 7 | +6 | **El Gran Remontador:** Verstappen estuvo muy activo en duelos directos, superando su consistencia en el resultado final. |
| **OCO** | 0.073232 | 2 | 8.67 | 11 | +9 | **Rey de la Zona Media:** Ocon lideró la actividad del midfield, con un PageRank inflado por la cantidad y calidad de sus rivales. |
| **BOR** | 0.061600 | 3 | 9.00 | 12 | +9 | Muy involucrado en batallas de la zona media-baja. |
| **LEC** | 0.055386 | 4 | 2.33 | 4 | 0 | Leclerc mantiene una correlación perfecta entre su velocidad final y su combatividad. |
| **PER** | 0.055219 | 5 | 12.33 | 15 | +10 | Pérez batalló a menudo en tráfico debido a malas clasificaciones, subiendo su PageRank. |
| **ANT** | 0.051754 | 6 | 2.00 | 2 | -4 | Antonelli rodó usualmente al frente, por lo que su PageRank es menor que su consistencia en carrera. |
| **LIN** | 0.051256 | 7 | 9.00 | 13 | +6 | Mantiene combatividad en el fondo de los puntos. |
| **ALB** | 0.051175 | 8 | 13.50 | 16 | +8 | Frecuentemente atrapado en tráfico, batallando en el midfield. |
| **NOR** | 0.050327 | 9 | 4.50 | 5 | -4 | Norris suele clasificar bien, lo que limita su volumen de adelantamientos en tráfico. |
| **BEA** | 0.045920 | 10 | 7.00 | 9 | -1 | Buena combatividad en el midfield superior. |
| **HAM** | 0.044105 | 13 | 2.00 | 3 | -10 | Hamilton rodó al frente del pelotón, teniendo pocas interacciones en el núcleo de combate. |
| **RUS** | 0.040306 | 15 | 1.00 | 1 | -14 | **Líder Consistente:** Russell lideró el campeonato promediando el 1er puesto, pero al no batallar en tráfico, su PageRank estructural es muy bajo. |

### C. Comparación con Modelos Predictivos y Rendimiento Físico

1.  **Modelos de Ritmo Físico vs. Grafo de Adelantamientos:**
    *   En modelos basados en ritmo físico (como la tasa de degradación de neumáticos a 3 vueltas `deg_rate_3lap` o ritmo puro de carrera en aire limpio), pilotos como **Russell**, **Leclerc** o **Hamilton** suelen liderar debido a la eficiencia aerodinámica e ingeniería de sus monoplazas. 
    *   Sin embargo, el **Grafo de Adelantamientos** califica una dimensión distinta: la **combatividad y la dominancia táctica en tráfico**. Un piloto en un coche rápido que sale en la pole y se escapa (ej. Russell) tendrá un ritmo físico impecable pero un PageRank bajo en esa carrera. En cambio, un piloto que califica mal y remonta (ej. Norris o Verstappen) acumulará una transferencia masiva de prestigio en el grafo. Así, los modelos físicos predicen velocidad pura, mientras que el PageRank mapea la eficacia en combate directo rueda a rueda.

---

## 5. Nota de Interpretación Deportiva (Fórmula 1)

El análisis estructural de grafos permite traducir conceptos tácticos de la F1 en propiedades topológicas de la red:

### A. Trenes de DRS ( DRS Trains) y Centralidad de Intermediación
En F1, un "Tren de DRS" ocurre cuando un monoplaza con baja velocidad punta o problemas de ritmo defiende su posición y lidera un grupo de varios coches rápidos que no pueden adelantar debido al efecto aerodinámico.
*   En la teoría de grafos, esto se manifiesta como una **alta Centralidad de Intermediación (Betweenness Centrality)**. El piloto que hace el "tapón" conecta los flujos de adelantamiento de los pilotos que vienen detrás.
*   **Ejemplo en Australia 2026:** **Carlos Sainz (SAI)** obtuvo una intermediación altísima de **0.187579** (liderando con creces esta métrica en la carrera), a pesar de que su PageRank fue de 0.069 (3º). Esto indica matemáticamente que Sainz lideró un denso pelotón y actuó como el cuello de botella del GP. **Franco Colapinto (COL)** con **0.137897** de intermediación representó el segundo gran tapón en la zona media de la parrilla.

### B. Líderes Fugados y Pilotos Aislados (Sinks & Isolated Nodes)
Un piloto dominante que escapa al frente de la carrera no adelanta a nadie (grado de entrada = 0) y no es adelantado (grado de salida = 0).
*   En la carrera individual, este piloto aparece como una **componente conexa aislada** o un nodo sumidero de tamaño mínimo.
*   **Ejemplo en Australia 2026:** **Oscar Piastri (PIA)** y **Nico Hülkenberg (HUL)** quedaron en componentes conexas de tamaño 1 (completamente aislados). Esto significa que corrieron en aire limpio o en tierra de nadie, sin realizar maniobras de adelantamiento ni defenderse activamente en pista, lo que explica su desconexión de la red de combate.

### C. El Pelotón Medio (*Midfield*) como Núcleo Denso
El *midfield* de la F1 es un ecosistema densamente conectado donde el rendimiento de los coches es similar. Esto se traduce en un subgrafo central grande y complejo (como la Componente Conexa de 16 pilotos en Australia), con caminos cíclicos (A adelanta a B, B adelanta a C, C adelanta a A). La alta densidad de enlaces y la reciprocidad en este núcleo explican por qué la centralidad se distribuye de manera homogénea y por qué pilotos como Ocon (`OCO`) o Bortoleto (`BOR`) registran un gran volumen de interacciones globales a lo largo de la temporada.
