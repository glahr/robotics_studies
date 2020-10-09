#!/usr/bin/env python3
from mujoco_py import load_model_from_xml, MjSim, MjViewer,load_model_from_path


model = load_model_from_path("./assets/two_link.xml")

sim = MjSim(model)
viewer = MjViewer(sim)

t = 0

try:
    while True:
        viewer.render()
        t += 1
        # if t > 500:
        #     break
except KeyboardInterrupt:
    print("saindo")
