from src.env import MorphologyAntEnv
from src.controllers import FixedCPGController
import numpy as np


env = MorphologyAntEnv()

obs, _ = env.reset(
    seed=42,
    intention=0,
)

controller = FixedCPGController()
controller.reset()

start = env.unwrapped.data.qpos[:2].copy()

fallen = False
total_reward = 0.0
steps = 0

print("Starting position:", start)
print("Running 500 CPG steps...")

for step in range(500):

    action = controller.act(
        intention=0,
        step=step,
        obs=obs,
    )

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    total_reward += reward
    steps += 1

    if info.get("fallen", False):
        fallen = True
        break

    if terminated or truncated:
        break


end = env.unwrapped.data.qpos[:2].copy()

distance = np.linalg.norm(
    end - start
)

print()
print("========== CPG TEST ==========")
print("Steps:", steps)
print("Start position:", start)
print("End position:", end)
print("Distance:", distance)
print("Total reward:", total_reward)
print("Fallen:", fallen)
print("================================")

env.close()