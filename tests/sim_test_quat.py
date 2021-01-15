#!/usr/bin/env python3
from mujoco_py import load_model_from_xml, MjSim, MjViewer, load_model_from_path
import mujoco_py
import numpy as np
import matplotlib.pyplot as plt

model = load_model_from_path("./assets/box.xml")

sim = MjSim(model)
viewer = MjViewer(sim)

t = 0

try:
    while True:
        # sim.data.ctrl[:] = 1
        # sim.step()
        sim.forward()
        viewer.render()
        t += 1

except KeyboardInterrupt:
    print("saindo")
