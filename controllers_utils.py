import numpy as np
import mujoco_py
import matplotlib.pyplot as plt


class CtrlUtils:

    def __init__(self, sim_handle, simulation_time, use_gravity, plot_2d, use_kd, use_ki):
        self.sim = sim_handle
        self.dt = sim_handle.model.opt.timestep
        self.use_gravity = use_gravity
        self.plot_2d = plot_2d
        self.mass_links = self.sim.model.body_mass[4:11]
        self.name_bodies = [self.sim.model.body_id2name(i) for i in range(4, 11)]
        self.name_tcp = self.sim.model.site_id2name(1)
        self.name_ft_sensor = self.sim.model.site_id2name(2)

        self.n_timesteps = int(simulation_time / self.dt)

        # max_diag_element = 0
        self.error_q_ant = 0
        # self.erro_q = q_ref[0] - sim.data.qpos
        self.q_log = np.zeros((self.n_timesteps, sim_handle.model.nv))
        self.time_log = np.zeros((self.n_timesteps, 1))
        self.H = np.zeros(self.sim.model.nv * self.sim.model.nv)  # inertia matrix
        self.qd = np.zeros((self.sim.model.nv,))

        self.controller_type = 'independent_joints'

        # Definition of step vectors.
        self.q_ref = [np.zeros((7,)) for _ in range(self.n_timesteps)]
        self.qvel_ref = [np.zeros((7,)) for _ in range(self.n_timesteps)]
        self.qacc_ref = [np.zeros((7,)) for _ in range(self.n_timesteps)]
        self.error_q = 0
        self.error_qvel = 0
        self.error_q_int_ant = 0
        self.error_q_ant = 0
        self.Kp = None
        self.Kd = None
        self.Ki = None
        self.use_kd = use_kd
        self.use_ki = use_ki
        self.lambda_H = 0
        self.kp = 0

    def trajectory_gen_joints(self, qd, tf, ti=0, dt=0.002, q_act=np.zeros((7,)), traj=None):
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

        self.q_ref = [qd for _ in range(self.n_timesteps)]

        time = np.linspace(ti, tf, int((tf - ti) / dt))

        if traj == 'spline3':
            T = tf - ti
            q0 = q_act
            qvel0 = np.zeros((7,))
            qf = qd
            qvelf = np.zeros((7,))
            Q = np.array([q0, qvel0, qf, qvelf])
            A_inv = np.linalg.inv(np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [1, T, T ** 2, T ** 3],
                                            [0, 1, 2 * T, 3 * T ** 2]]))
            coeffs = np.dot(A_inv, Q)
            for i, t in enumerate(time):
                self.q_ref[i] = np.dot(np.array([1, (t - ti), (t - ti) ** 2, (t - ti) ** 3]), coeffs)
                self.qvel_ref[i] = np.dot(np.array([0, 1, 2 * (t - ti), 3 * (t - ti) ** 2]), coeffs)
                self.qacc_ref[i] = np.dot(np.array([0, 0, 2, 6 * (t - ti)]), coeffs)

        if traj == 'spline5':
            T = tf - ti
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
                                            [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
                                            [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                                            [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]]))
            coeffs = np.dot(A_inv, Q)
            for i, t in enumerate(time):
                self.q_ref[i] = np.dot(np.array([1, (t - ti), (t - ti) ** 2, (t - ti) ** 3, (t - ti) ** 4, (t - ti) ** 5]),
                                  coeffs)
                self.qvel_ref[i] = np.dot(
                    np.array([0, 1, 2 * (t - ti), 3 * (t - ti) ** 2, 4 * (t - ti) ** 3, 5 * (t - ti) ** 4]), coeffs)
                self.qacc_ref[i] = np.dot(np.array([0, 0, 2, 6 * (t - ti), 12 * (t - ti) ** 2, 20 * (t - ti) ** 3]), coeffs)

        # return q_ref, qvel_ref, qacc_ref

    def ctrl_independent_joints(self):
        error_q_int = (self.error_q + self.error_q_ant) * self.dt / 2 + self.error_q_int_ant
        self.error_q_int_ant = error_q_int
        return np.dot(self.Kp, self.error_q) + np.dot(self.Kd, self.error_qvel) + np.dot(self.Ki, error_q_int)

    def ctrl_action(self, sim):
        if self.controller_type == 'independent_joints':
            if self.Kp is None:
                self.get_pd_matrices()
            u = self.ctrl_independent_joints()

        if self.use_gravity:
            u += self.tau_g(sim)
        return u

    def calculate_errors(self, sim_handle, k, qd_new=0):
        qpos = sim_handle.data.qpos
        qvel = sim_handle.data.qvel
        self.error_q = self.q_ref[k] + qd_new - qpos
        self.error_qvel = self.qvel_ref[k] - qvel

    def kuka_subtree_mass(self):
        body_names = ['kuka_link_{}'.format(i + 1) for i in range(7)]
        body_ids = [self.sim.model.body_name2id(n) for n in body_names]
        return self.sim.model.body_subtreemass[body_ids]

    def get_pd_matrices(self):
        subtree_mass = self.kuka_subtree_mass()
        Kp = np.eye(7)
        for i in range(7):
            Kp[i, i] = self.kp * subtree_mass[i]

        if self.use_kd:
            Kd = np.eye(7)
            for i in range(7):
                if i == 6:
                    Kd[i, i] = Kp[i, i] ** (0.005 / self.kp)
                else:
                    Kd[i, i] = Kp[i, i] ** 0.5
            # for i in range(7):
            #     if i == 6:
            #         Kd[i, i] = Kp[i, i] ** 0.005 * (7 - i)
            #     elif i == 4:
            #         Kd[i, i] = Kp[i, i] ** 0.1 * (10 - i)
            #     else:
            #         Kd[i, i] = Kp[i, i] ** 0.25 * (10 - i)
        else:
            Kd = np.zeros((7, 7))

        if self.use_ki:
            Ki = np.eye(7)
            for i in range(7):
                Ki[i, i] = Kp[i, i] * Kd[i, i] / self.lambda_H ** 2
        else:
            Ki = np.zeros((7, 7))
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def tau_g(self, sim):
        Jp_shape = (3, sim.model.nv)
        comp = np.zeros((sim.model.nv,))
        for body, mass in zip(self.name_bodies, self.mass_links):
            Jp = sim.data.get_body_jacp(body).reshape(Jp_shape)
            comp = comp - np.dot(Jp.transpose(), sim.model.opt.gravity * mass)
        return comp

    def plots(self):
        plt.plot(self.time_log, self.q_log)
        plt.plot(self.time_log, [q_r for q_r in self.q_ref], 'k--')
        plt.legend(['q' + str(i + 1) for i in range(7)])
        plt.show()

    def step(self, sim, k):
        if self.plot_2d:
            self.q_log[k] = sim.data.qpos
            self.time_log[k] = sim.data.time
        self.error_q_ant = self.error_q
        sim.step()

    # def dynamic_values(self):
    #     # inertia matrix H
    #     mujoco_py.functions.mj_fullM(sim.model, H, sim.data.qM)
    #     H_ = H.reshape(sim.model.nv, sim.model.nv)
    #     element = max(np.diag(H_))
    #     if element > max_diag_element:
    #         max_diag_element = element
    #     # internal forces: Coriolis + gravitational
    #     C = np.zeros(7)
    #     C = sim.data.qfrc_bias
