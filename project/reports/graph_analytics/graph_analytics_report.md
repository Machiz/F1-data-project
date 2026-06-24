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

## 4. Sección Comparativa: PageRank vs. Popularidad y Modelos

### Tabla Comparativa: Dominancia (PageRank) vs. Popularidad (Adelantamientos Totales)

| Piloto | Adelantamientos (Ofensiva) | Rank Popularidad | PageRank (Dominancia) | Rank PageRank | Diferencia de Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **VER** | 46 | 1 | 0.073675 | 1 | 0 |
| **OCO** | 45 | 2 | 0.073232 | 2 | 0 |
| **BOR** | 37 | 3 | 0.061600 | 3 | 0 |
| **LEC** | 33 | 4 | 0.055386 | 4 | 0 |
| **PER** | 33 | 4 | 0.055219 | 5 | -1 |
| **ANT** | 30 | 7 | 0.051754 | 6 | +1 |
| **LIN** | 28 | 9 | 0.051256 | 7 | +2 |
| **ALB** | 31 | 6 | 0.051175 | 8 | -2 |
| **NOR** | 30 | 7 | 0.050327 | 9 | -2 |
| **BEA** | 27 | 10 | 0.045920 | 10 | 0 |

### Análisis de Diferencias y Comparación con Modelos

1.  **PageRank vs. Popularidad (Adelantamientos Hechos):**
    *   **Max Verstappen (VER)** y **Esteban Ocon (OCO)** lideran tanto en adelantamientos totales como en PageRank. Sin embargo, aunque Ocon realizó casi la misma cantidad de adelantamientos (45 vs 46), Verstappen mantiene la primera posición en PageRank. Esto ocurre porque los adelantamientos de Verstappen fueron de "mayor calidad" (hechos a pilotos competitivos en la parte alta), mientras que Ocon opera en el denso pelotón medio, donde la dominancia se diluye rápidamente debido a la alta tasa de adelantamientos recíprocos.
    *   **LIN** sube +2 posiciones en PageRank a pesar de tener solo 28 adelantamientos (9º en popularidad). Esto demuestra que sus pocos adelantamientos fueron sobre pilotos de alto nivel y con buena retención de posición.
    *   **ALB** y **NOR** caen -2 posiciones en PageRank a pesar de tener 31 y 30 adelantamientos respectivamente. Esto indica que un porcentaje significativo de sus adelantamientos fue sobre pilotos del fondo de la parrilla (bajo PageRank), aportando menos prestigio a su puntuación acumulada.

2.  **Comparación con Modelos Predictivos / Rendimiento Físico:**
    *   En modelos basados en ritmo físico (como la tasa de degradación de neumáticos a 3 vueltas `deg_rate_3lap` o ritmo puro de carrera), pilotos como **Leclerc** o **Norris** suelen liderar debido al rendimiento inherente del coche. 
    *   Sin embargo, el **Grafo de Adelantamientos** califica una dimensión distinta: la **eficacia en carrera y el combate directo**. Un piloto en un monoplaza rápido que sale en pole y se escapa (ej. Leclerc cuando domina un GP) tendrá muy pocos adelantamientos en pista, por lo que su PageRank en esa carrera será bajo, a pesar de que los modelos predictivos de ritmo lo califiquen como el más rápido. El PageRank, por ende, es una métrica de combatividad y dominancia táctica en tráfico, mientras que los modelos de ritmo miden eficiencia de ingeniería pura.

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
