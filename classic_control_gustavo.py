#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
from numpy import cos, pi, array
import os
import gym

model = load_model_from_path("assets/two_link.xml")

sim = MjSim(model)

viewer = MjViewer(sim)
# modder = TextureModder(sim)

t = 0

# sim.data.qpos[:] = array([0, 0, 0, -pi/2, 0, 0, 0])

while True:
    

    sim.data.ctrl[0] = 100*cos(0.01*t)
    sim.data.ctrl[1] = cos(0.01*t)

    sim.step()
    viewer.render()
    t += 1
    # if t > 100:  # and os.getenv('TESTING') is not None:
    #     break
