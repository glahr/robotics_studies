#!/usr/bin/env python3
from mujoco_py import load_model_from_xml, MjSim, MjViewer,load_model_from_path
import numpy as np

model = load_model_from_path("./assets/full_kuka_all_joints.xml")

sim = MjSim(model)
viewer = MjViewer(sim)

t = 0

kp = 1
kd = 1

Kp = np.eye(7)
Kd = np.eye(7)

for i in range(7):
    Kp[i, i] = (7-i)*kp
    Kd[i, i] = (7-i)*kd

qpos_d_final = np.array([-2.2116084396651, -0.782177335557374, 1.11375474368298, 1.38205849413128, -0.3207054916224, -2.0942, -1.97501802948687])
qpos_d_init = np.array([-1.92583208694633, 1.50069279816198, 2.42928397578649, 1.93362964508148, 0.418910404164455, 0.263363368399808, -1.97501802948687])
# qpos_d_final = [2.20849708880688, 0.779849599807444, 2.03531542176006, 1.37956779462408, 0.319538377656998, -2.0942, -2.73036507925815]
# qpos_d_init = [2.63990994384753, -0.0400318261139474, -0.368200292935267, 1.1419905993935, 1.44819338372141, -0.239140603636931, -2.73036507925815]
qvel_d = np.zeros((7,))

eps = 0.1

try:
    while True:
        qpos = sim.data.qpos[:]
        qvel = sim.data.qvel[:]

        qpos_error = qpos_d_init - qpos
        qvel_error = qvel_d - qvel

        u = np.dot(Kp, qpos_error) + np.dot(Kd, qvel_error)

        sim.data.ctrl[:] = u

        if np.all(np.absolute(qpos_error) < eps):
            qpos_d_init = qpos_d_final

        sim.step()

        viewer.render()
        t += 1
        # if t > 500:
        #     break
except KeyboardInterrupt:
    print("saindo")
