#!/usr/bin/env python3
from mujoco_py import load_model_from_xml, MjSim, MjViewer, load_model_from_path
import mujoco_py
import numpy as np
import matplotlib.pyplot as plt

model = load_model_from_path("./assets/full_kuka_all_joints_gravity.xml")

sim = MjSim(model)
viewer = MjViewer(sim)

t = 0
n_timesteps = int(10/0.002)

qpos_d = np.array([0, 0.461, 0, -0.817, 0, 0.69, 0])
qvel_d = np.zeros(7)

qlog = np.zeros((n_timesteps, sim.model.nv))

Kp = np.eye(7)
Kd = np.eye(7)

for i in range(sim.model.nv):
    Kp[i, i] = 10*(1-0.15*i)
    Kd[i, i] = 1.2*(Kp[i, i]/2)**0.5

Kd[6, 6] = Kd[6, 6]/50

qacc_d = np.zeros(7)
qvel_d = np.zeros(7)

H = np.zeros(7*7)

def comp_gravity(sim):
    name_bodies = [sim.model.body_id2name(i) for i in range(4, 11)]
    mass_links = sim.model.body_mass[4:11]
    Jp_shape = (3, sim.model.nv)
    comp = np.zeros((sim.model.nv,))
    for body, mass in zip(name_bodies, mass_links):
        Jp = sim.data.get_body_jacp(body).reshape(Jp_shape)
        comp = comp - np.dot(Jp.T, sim.model.opt.gravity * mass)
    return comp

try:
    while t < n_timesteps:

        qlog[t] = sim.data.qpos

        qvel = sim.data.qvel
        qpos = sim.data.qpos

        qpos_erro = qpos_d - qpos
        qvel_erro = qvel_d - qvel

        C_eq = sim.data.qfrc_bias  # C_eq = C*qvel + tau_g

        v = qacc_d + Kd.dot(qvel_d - qvel) + Kp.dot(qpos_d - qpos)

        mujoco_py.functions.mj_fullM(sim.model, H, sim.data.qM)

        u = np.reshape(H, (7, 7)).dot(v) + C_eq

        sim.data.ctrl[:] = u

        sim.step()
        viewer.render()
        t += 1

except KeyboardInterrupt:
    print("saindo")

plt.plot(qlog)
plt.plot([qpos_d for _ in range(n_timesteps)], 'k--')
plt.show()
