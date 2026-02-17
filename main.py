from src.core.engine import SimulationEngine
import src.utils.config as consts
from src.core.state import SimulationState
from src.utils.cosmology import generate_zeldovich_ics
import cupy as cp
import os

print("N-Body Simulation:\n")
print("-----------------------------------------\n")
print(f"Number of celestial bodies : {consts.N_BODIES}\n")
print(f"Number of steps : {consts.N_STEPS}\n")
print("------------------------------------------\n\n")
print("Initialization of Initial Conditions...\n")

initial_pos, initial_vel = generate_zeldovich_ics(consts)

print("Simulation Initialization...\n")

simulation_state = SimulationState(consts.N_BODIES, initial_pos, initial_vel)

physics_engine = SimulationEngine(consts, simulation_state)

physics_engine.run()

print("Simulation done!\n\n")
print("Writing frames to disk & making video (This may take a while)...\n")

os.makedirs(consts.OUTPUT_DIR, exist_ok=True) 
physics_engine.save()

print("Video done.")
