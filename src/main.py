from src.core.engine import SimulationEngine
import src.utils.config as consts
from src.core.state import SimulationState
from src.utils.cosmology import generate_zeldovich_ics
import cupy as cp

print("N-Body Simulation:\n")
print("-----------------------------------------\n")
print(f"Number of celestial bodies : {consts.N_BODIES}\n")
print(f"Number of steps : {consts.N_STEPS}\n")
print("------------------------------------------\n\n")
print("Initialization of Initial Conditions...\n")

initial_pos, initial_vel = generate_zeldovich_ics(consts)
px_d, py_d, pz_d = initial_pos[0], initial_pos[1], initial_pos[2]
vx_d, vy_d, vz_d = initial_vel[0], initial_vel[1], initial_vel[2]

print("Simulation Initialization...\n")

simulation_state = SimulationState(consts.N_BODIES)
physics_engine = SimulationEngine(consts, simulation_state)

physics_engine.run()

print("Simulation done!\n\n")
print("Writing frames to disk (This may take a while)...\n")

physics_engine.save()

print("Writing Video...")