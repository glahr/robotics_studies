#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
from numpy import cos, pi, array
import numpy as np

simulate = True

model = load_model_from_path("/home/glahr/mujoco_gym/gym-kuka-mujoco/gym_kuka_mujoco/envs/assets/full_kuka_all_joints.xml")

sim = MjSim(model)

if simulate:
    viewer = MjViewer(sim)

q_ref = np.array([0, 0.461, 0, -0.817, 0, 0.69, 0])
qvel_ref = np.array([0, 0, 0, 0, 0, 0, 0])
# q_log = []

kp = 1
kd = 1
Kp = np.eye(7)
Kd = np.eye(7)
Kd = np.eye(7)
eps = 0.05

for i in range(7):
    Kp[i, i] = kp*(7-i)

for i in range(7):
    Kd[i, i] = kp**0.5*(7-i)

# sim.data.qpos[:] = array([0, 0, 0, -pi/2, 0, 0, 0])

while True:

    qpos = sim.data.qpos
    qvel = sim.data.qvel
    erro_q = q_ref - qpos
    erro_v = qvel_ref - qvel

    J = np.concatenate((sim.data.site_jacp[-1].reshape((7, -1)).transpose(), sim.data.site_jacr[-1].reshape((7, -1)).transpose()))

    J_pseudo = np.linalg.pinv(J)

    if sim.data.time > 5 and abs(erro_q[1]) < eps:
        erro_q = np.dot(Kd, np.dot(np.eye(7) - np.dot(J_pseudo, J), erro_q))

    u = np.dot(Kp, erro_q) + np.dot(Kd, erro_v)

    sim.data.ctrl[:] = u

    # if (np.absolute(erro_q).all() < eps).all():
    # if abs(erro_q[1]) < eps:
    #     print("tolerancia " + str(sim.data.time))

    sim.step()
    if simulate:
        viewer.render()
    # t += 1
    # if t > 100:  # and os.getenv('TESTING') is not None:
    #     break
