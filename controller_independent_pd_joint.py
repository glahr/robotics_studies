#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
from gym_kuka_mujoco.utils.kinematics import forwardKin, forwardKinJacobian, forwardKinSite, forwardKinJacobianSite
import mujoco_py
import matplotlib.pyplot as plt
import numpy as np


def trajectory_gen_joints(qd, tf, n, ti=0, dt=0.002, q_act=np.zeros((7,)), traj=None):
    q_ref = [qd for _ in range(n_timesteps)]
    qvel_ref = [np.array([0, 0, 0, 0, 0, 0, 0]) for _ in range(n_timesteps)]
    # q_d = q_ref[-1]
    time = np.linspace(ti, tf, int((tf - ti) / dt))

    # T = tf-ti
    # qd_ti = q_act
    # qveld_ti = np.zeros((7,1))
    # # qd_tf = qd[-1]
    # qveld_tf = np.zeros((7,1))
    # # q_ = [np.array([qd_ti[i], qveld_ti[i], qd_tf, qveld_tf[i]]) for i, qd_tf in enumerate(qd)]
    # 
    # A = np.array([[1, 0, 0, 0],
    #               [0, 1, 0, 0],
    #               [1, T, T**2, T**3],
    #               [0, 1, 2*T,  3*T**2]])
    # 
    # for i, qd_ti_i, qveld_ti_i, qd_tf_i, qveld_tf_i in zip()
    # 
    # a, b, c, d = [np.dot(np.linalg.inv(A), q) for q in q_]
    # 
    # q_ref = np.array([a+b*(t-ti)+c*(t-ti)**2+d*(t-ti)**3 for t in time])

    if traj == 'spline':
        for i, t in enumerate(time):
            q_ref[i] = 2 * (q_act - qd) / tf ** 3 * t ** 3 - 3 * (q_act - qd) / tf ** 2 * t ** 2 + q_act
            qvel_ref[i] = 6 * (q_act - qd) / tf ** 3 * t ** 2 - 6 * (q_act - qd) / tf ** 2 * t
            # qacc_ref[i] = 12 * (q0 - q_d) / tf ** 3 * t - 6 * (q0 - q_d) / tf ** 2
    return q_ref, qvel_ref


def ctrl_independent_joints(error_q, error_v, error_q_int_ant):
    dt = 0.002
    error_q_int = (error_q + error_q_ant)*dt/2 + error_q_int_ant
    error_q_int_ant = error_q_int
    return np.dot(Kp, erro_q) + np.dot(Kd, erro_v) + np.dot(Ki, error_q_int)


def get_ctrl_action(controller, error_q, error_v, error_q_int_ant=0):
    if controller == 'independent_joints':
        u = ctrl_independent_joints(error_q, error_v, error_q_int_ant)
    return u, error_q_int_ant


def kuka_subtree_mass():
    body_names = ['kuka_link_{}'.format(i + 1) for i in range(7)]
    body_ids = [sim.model.body_name2id(n) for n in body_names]
    return sim.model.body_subtreemass[body_ids]


def get_pd_matrices(kp, use_ki, use_kd, lambda_H=0):
    subtree_mass = kuka_subtree_mass()
    Kp = np.eye(7)
    for i in range(7):
        Kp[i, i] = kp * subtree_mass[i]

    if use_kd:
        Kd = np.eye(7)
        for i in range(7):
            if i == 6:
                Kd[i, i] = Kp[i, i]**(0.005/kp)
            else:
                Kd[i, i] = 2*Kp[i, i]**0.5
        # for i in range(7):
        #     if i == 6:
        #         Kd[i, i] = Kp[i, i] ** 0.005 * (7 - i)
        #     elif i == 4:
        #         Kd[i, i] = Kp[i, i] ** 0.1 * (10 - i)
        #     else:
        #         Kd[i, i] = Kp[i, i] ** 0.25 * (10 - i)
    else:
        Kd = np.zeros((7,7))

    if use_ki:
        Ki = np.eye(7)
        for i in range(7):
            Ki[i, i] = Kp[i,i]*Kd[i,i]/lambda_H**2
    else:
        Ki = np.zeros((7,7))
    return Kp, Kd, Ki


if __name__ == '__main__':

    simulate = True

    model = load_model_from_path("assets/full_kuka_all_joints.xml")

    sim = MjSim(model)

    if simulate:
        viewer = MjViewer(sim)

    tf = 1
    n_timesteps = 3000
    controller_type = 'independent_joints'
    if controller_type == 'independent_joints':
        kp = 1.5
        use_kd = True
        use_ki = False
        lambda_H = 6.2/sim.model.nv
        error_q_int_ant = 0
        Kp, Kd, Ki = get_pd_matrices(kp=kp, use_ki=use_ki, use_kd=use_kd, lambda_H=lambda_H)
    k = 1

    qd = np.array([0, 0.461, 0, -0.817, 0, 0.69, 0])
    # qd = np.array([0, 0, 0, 0, 0, 0, 0])
    # qd = np.array([0, 0, 0, -np.pi/2, -np.pi/2, 0, 0])
    q_ref, qvel_ref = trajectory_gen_joints(qd, tf, n_timesteps, traj='spline')
    q_log = np.zeros((n_timesteps, 7))
    time_log = np.zeros((n_timesteps, 1))
    H = np.zeros(sim.model.nv * sim.model.nv)

    eps = 1*np.pi/180

    mass_links = sim.model.body_mass[4:11]
    name_body = [sim.model.body_id2name(i) for i in range(4, 11)]
    name_tcp = sim.model.site_id2name(1)
    name_ft_sensor = sim.model.site_id2name(2)
    jac_shape = (3, sim.model.nv)
    C = np.zeros(7)

    sim.forward()

    max_diag_element = 0
    error_q_ant = 0
    erro_q = q_ref[0] - sim.data.qpos
    qd = np.array([0, 0, 0, 0, 0, 0, 0])

    while True:

        # if (np.absolute(erro_q) < eps).all():
        #     qd = np.array([0, 0, 3/2*np.pi/2, 0, 0, -np.pi/2, 0])
        # print("tolerancia " + str(sim.data.time))

        qpos = sim.data.qpos
        qvel = sim.data.qvel
        erro_q = q_ref[k] + qd - qpos
        erro_v = qvel_ref[k] - qvel

        # inertia matrix H
        mujoco_py.functions.mj_fullM(sim.model, H, sim.data.qM)
        H_ = H.reshape(sim.model.nv, sim.model.nv)
        element = max(np.diag(H_))
        if element > max_diag_element:
            max_diag_element = element
        # internal forces: Coriolis + gravitational
        C = sim.data.qfrc_bias

        u, error_q_int_ant = get_ctrl_action(controller_type, erro_q, erro_v, error_q_int_ant)

        sim.data.ctrl[:] = u

        sim.step()
        if simulate:
            viewer.render()
        k += 1
        if k >= n_timesteps:  # and os.getenv('TESTING') is not None:
            break

        q_log[k] = qpos
        time_log[k] = sim.data.time
        error_q_ant = erro_q

    plt.plot(time_log, q_log)
    plt.plot(time_log, [q_r for q_r in q_ref+qd], 'k--')
    plt.legend(['q'+str(i+1) for i in range(7)])
    plt.show()
    print("biggest element = ", max_diag_element)
