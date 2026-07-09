# 🏁 F1 Tactical Strategy Assistant - Interactive Chatbot Demo

This demo provides a modular, interactive terminal chatbot that simulates the F1 strategist's Pit Wall briefing. By leveraging the Layer 1 Stacking Regressor and Layer 2 Point-wise Ranker predictions, the assistant translates F1 telemetry and race graphs into strategic recommendations.

---

## 🛠️ How to Launch the Demo

From the root of the `project/` directory, execute:

```bash
python demo/run_demo.py
```

---

## 🏎️ Available Commands inside the Chat

*   `list` — Shows all available races and active driver acronyms in the dataset.
*   `help` — Reprints the welcome screen.
*   `exit` — Closes the strategy box communication channel.

---

## 🚥 Sample Strategic Scenarios to Try

Try running these query formats: `[carrera] [piloto] [vuelta]`.

1.  **United States GP — Verstappen (`VER`) on Lap 39:**
    ```text
    [PitWall-IA] > united_states VER 39
    ```
    *Insight:* Check the suggested pits priority table for `wait_laps` to see if extending the stint is optimal.

2.  **Australia GP — Hamilton (`HAM`) on Lap 25:**
    ```text
    [PitWall-IA] > australia HAM 25
    ```
    *Insight:* Inspect the physical tire degradation cost (Capa 1) and traffic warnings behind.

3.  **China GP — Norris (`NOR`) on Lap 15:**
    ```text
    [PitWall-IA] > china NOR 15
    ```
    *Insight:* Check the Midfield train Betweenness Centrality warning from graph analytics.
