#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
from gym_kuka_mujoco.utils.kinematics import forwardKin, forwardKinJacobian, forwardKinSite, forwardKinJacobianSite
import mujoco_py
import matplotlib.pyplot as plt
import numpy as np


def trajectory_gen_joints(qd, tf, n, ti=0, dt=0.002, q_act=np.zeros((7,)), traj=None):
    """
    Joint trajectory generation with different methods.
    :param qd: joint space desired final point
    :param tf: total time of travel
    :param n: number of time steps
    :param ti: initial time
    :param dt: time step
    :param q_act: current joint position
    :param traj: type of trajectory 'step', 'spline3', 'spline5', 'trapvel'
    :return:
    """

    # Definition of step vectors.
    q_ref = [qd for _ in range(n_timesteps)]
    qvel_ref = [np.zeros((7,)) for _ in range(n_timesteps)]
    qacc_ref = [np.zeros((7,)) for _ in range(n_timesteps)]
    # q_d = q_ref[-1]
    time = np.linspace(ti, tf, int((tf - ti) / dt))

    if traj == 'spline3':
        T = tf-ti
        q0 = q_act
        qvel0 = np.zeros((7,))
        qf = qd
        qvelf = np.zeros((7,))
        Q = np.array([q0, qvel0, qf, qvelf])
        A_inv = np.linalg.inv(np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [1, T, T**2, T**3],
                                        [0, 1, 2*T,  3*T**2]]))
        coeffs = np.dot(A_inv, Q)
        for i, t in enumerate(time):
            q_ref[i] = np.dot(np.array([1, (t - ti), (t - ti)**2, (t - ti)**3]), coeffs)
            qvel_ref[i] = np.dot(np.array([0, 1, 2*(t - ti), 3*(t - ti) ** 2]), coeffs)
            qacc_ref[i] = np.dot(np.array([0, 0, 2, 6 * (t - ti)]), coeffs)

    if traj == 'spline5':
        T = tf-ti
        q0 = q_act
        qvel0 = np.zeros((7,))
        qacc0 = np.zeros((7,))
        qf = qd
        qvelf = np.zeros((7,))
        qaccf = np.zeros((7,))
        Q = np.array([q0, qvel0, qacc0, qf, qvelf, qaccf])
        A_inv = np.linalg.inv(np.array([[1, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0],
                                        [0, 0, 2, 0, 0, 0],
                                        [1, T, T**2, T**3, T**4, T**5],
                                        [0, 1, 2*T,  3*T**2, 4*T**3, 5*T**4],
                                        [0, 0, 2,  6*T, 12*T**2, 20*T**3]]))
        coeffs = np.dot(A_inv, Q)
        for i, t in enumerate(time):
            q_ref[i] = np.dot(np.array([1, (t - ti), (t - ti)**2, (t - ti)**3, (t - ti)**4, (t - ti)**5]), coeffs)
            qvel_ref[i] = np.dot(np.array([0, 1, 2*(t - ti), 3*(t - ti) ** 2, 4*(t - ti)**3, 5*(t - ti)**4]), coeffs)
            qacc_ref[i] = np.dot(np.array([0, 0, 2, 6 * (t - ti), 12*(t - ti)**2, 20*(t - ti)**3]), coeffs)

    return q_ref, qvel_ref, qacc_ref


def ctrl_independent_joints(error_q, error_v, error_q_int_ant):
    dt = 0.002
    error_q_int = (error_q + error_q_ant)*dt/2 + error_q_int_ant
    error_q_int_ant = error_q_int
    return np.dot(Kp, erro_q) + np.dot(Kd, erro_v) + np.dot(Ki, error_q_int)


def get_ctrl_action(controller, error_q, error_v, error_q_int_ant=0):
    if controller == 'independent_joints':
        u = ctrl_independent_joints(error_q, error_v, error_q_int_ant)
    return u


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
                Kd[i, i] = Kp[i, i]**0.5
        # for i in range(7):
        #     if i == 6:
        #         Kd[i, i] = Kp[i, i] ** 0.005 * (7 - i)
        #     elif i == 4:
        #         Kd[i, i] = Kp[i, i] ** 0.1 * (10 - i)
        #     else:
        #         Kd[i, i] = Kp[i, i] ** 0.25 * (10 - i)
    else:
        Kd = np.zeros((7, 7))

    if use_ki:
        Ki = np.eye(7)
        for i in range(7):
            Ki[i, i] = Kp[i, i]*Kd[i, i]/lambda_H**2
    else:
        Ki = np.zeros((7,7))
    return Kp, Kd, Ki


def tau_g(sim, name_bodies, mass_links):
    Jp_shape = (3, sim.model.nv)
    comp = np.zeros((sim.model.nv,))
    for body, mass in zip(name_bodies, mass_links):
        Jp = sim.data.get_body_jacp(body).reshape(Jp_shape)
        comp = comp - np.dot(Jp.transpose(), sim.model.opt.gravity*mass)
    return comp


if __name__ == '__main__':

    simulate = True
    use_gravity = True
    model_name = "assets/full_kuka_all_joints"
    if use_gravity:
        model_name += "_gravity"

    model_xml = load_model_from_path(model_name + '.xml')

    sim = MjSim(model_xml)

    if simulate:
        viewer = MjViewer(sim)

    tf = 1  # spline final time
    n_timesteps = 3000
    controller_type = 'independent_joints'
    if controller_type == 'independent_joints':
        kp = 7
        use_kd = True
        use_ki = False
        lambda_H = 6.2/sim.model.nv
        error_q_int_ant = 0
        Kp, Kd, Ki = get_pd_matrices(kp=kp, use_ki=use_ki, use_kd=use_kd, lambda_H=lambda_H)
    k = 1

    qd = np.array([0, 0.461, 0, -0.817, 0, 0.69, 0])
    # qd = np.array([0, 0, 0, 0, 0, 0, 0])
    # qd = np.array([0, 0, 0, -np.pi/2, -np.pi/2, 0, 0])
    q_ref, qvel_ref, qacc_ref = trajectory_gen_joints(qd, tf, n_timesteps, traj='spline5')
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

        u = get_ctrl_action(controller_type, erro_q, erro_v, error_q_int_ant=error_q_int_ant)

        if use_gravity:
            u += tau_g(sim, name_body, mass_links)
        # u = tau_g(sim.model, sim.data, name_body, mass_links)

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
