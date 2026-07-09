import sys
import time
import threading
from realtime_pipeline import RealtimePipeline
from realtime_render import render_dashboard

class RealtimeSimulation:
    def __init__(self, race_name="united_kingdom", driver_acronym="VER"):
        self.race_name = race_name
        self.driver_acronym = driver_acronym
        self.pipeline = RealtimePipeline(race_name, driver_acronym)
        
        self.current_lap = 1
        self.total_laps = 0
        self.speed_multiplier = 1.0
        self.is_paused = False
        self.running = False
        
        self.next_lap_event = threading.Event()
        self.quit_event = threading.Event()
        
    def start(self):
        """Starts resource loading and launches simulation and input threads."""
        try:
            self.pipeline.load_resources()
        except Exception as e:
            print(f"\033[1;31m[ERROR] No se pudieron cargar los recursos: {e}\033[0m")
            return
            
        self.total_laps = self.pipeline.get_total_laps()
        self.current_lap = 3  # Start at lap 3 to ensure we have at least 3 laps of history for rolling metrics
        
        self.running = True
        
        # 1. Start Input Thread
        input_thread = threading.Thread(target=self._input_loop, daemon=True)
        input_thread.start()
        
        # 2. Run Simulation Loop in Main Thread
        self._simulation_loop()
        
    def _simulation_loop(self):
        """Main loop that steps lap by lap, runs live inference, and renders the dashboard."""
        while self.running and not self.quit_event.is_set():
            if self.current_lap > self.total_laps:
                print("\n[FINISHED] ¡Bandera a cuadros! La carrera ha finalizado.")
                break
                
            # Perform real-time inference (no future data used)
            candidates = self.pipeline.get_realtime_inference(self.current_lap)
            leaders = self.pipeline.get_leaderboard_at_lap(self.current_lap)
            
            # Render dashboard
            render_dashboard(
                candidates_df=candidates,
                race=self.race_name,
                driver=self.driver_acronym,
                lap=self.current_lap,
                leaders=leaders,
                total_laps=self.total_laps,
                speed_multiplier=self.speed_multiplier,
                is_paused=self.is_paused
            )
            
            # Reset event before waiting
            self.next_lap_event.clear()
            
            # Wait for timeout OR event trigger (instant transition)
            while self.running and not self.quit_event.is_set():
                if self.is_paused:
                    time.sleep(0.2) # Sleep shortly while paused
                    continue
                    
                sleep_time = 3.0 / self.speed_multiplier
                triggered = self.next_lap_event.wait(timeout=sleep_time)
                
                if triggered:
                    # Lap skipped instantly or command entered
                    self.next_lap_event.clear()
                break
                
            if not self.is_paused:
                self.current_lap += 1
                
        self.running = False
        print("\n[BOX] Canal de simulacion en vivo cerrado. ¡Buen GP!")
        
    def _input_loop(self):
        """Thread loop that reads strategists command line inputs from standard input."""
        while self.running and not self.quit_event.is_set():
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                cmd = line.strip().lower()
                
                if cmd == "":
                    # Empty enter -> Advance lap instantly
                    self.next_lap_event.set()
                elif cmd in ["q", "quit", "exit"]:
                    self.quit_event.set()
                    self.next_lap_event.set()
                    break
                elif cmd in [" ", "space", "espacio"]:
                    # Toggle speed 1x / 2x
                    self.speed_multiplier = 2.0 if self.speed_multiplier == 1.0 else 1.0
                    self.next_lap_event.set() # Trigger re-render
                elif cmd in ["p", "pause", "pausa"]:
                    self.is_paused = not self.is_paused
                    self.next_lap_event.set() # Trigger re-render
                else:
                    # Allow simulating compound changes if desired
                    # e.g., 'change compound hard' (Optional extra controls)
                    if cmd.startswith("change compound"):
                        parts = cmd.split()
                        if len(parts) >= 3:
                            comp = parts[2].upper()
                            # We can update the compound ordinal in the pipeline (optional feature)
                            print(f"\n[INFO] Solicitando simulación de compuesto {comp}...")
                            time.sleep(1.0)
                    self.next_lap_event.set() # Trigger re-render
            except Exception:
                break
