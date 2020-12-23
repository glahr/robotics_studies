import numpy as np
import mujoco_py
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
import spatialmath as smath


class CtrlUtils:

    def __init__(self, sim_handle, simulation_time, use_gravity, plot_2d, use_kd, use_ki, controller_type):
        self.sim = sim_handle
        self.dt = sim_handle.model.opt.timestep
        self.use_gravity = use_gravity
        self.plot_2d = plot_2d
        self.mass_links = self.sim.model.body_mass[4:11]
        self.name_bodies = [self.sim.model.body_id2name(i) for i in range(4, 11)]
        self.name_tcp = self.sim.model.site_id2name(1)
        self.name_ft_sensor = self.sim.model.site_id2name(2)

        self.n_timesteps = int(simulation_time / self.dt)

        self.time_log = np.zeros((self.n_timesteps, 1))
        self.H = np.zeros(self.sim.model.nv * self.sim.model.nv)  # inertia matrix
        self.C = np.zeros((7,))  # Coriolis vector
        self.qd = np.zeros((self.sim.model.nv,))

        self.controller_type = controller_type

        # Definition of step vectors.
        if True: # controller_type == 'independent_joints' or controller_type == 'inverse_dynamics':
            self.q_ref = np.zeros((self.n_timesteps, sim_handle.model.nv))
            self.qvel_ref = np.zeros((self.n_timesteps, sim_handle.model.nv)) #[np.zeros((7,)) for _ in range(self.n_timesteps)]
            self.qacc_ref = np.zeros((self.n_timesteps, sim_handle.model.nv)) #[np.zeros((7,)) for _ in range(self.n_timesteps)]
            self.q_log = np.zeros((self.n_timesteps, sim_handle.model.nv))
            self.error_q = 0
            self.error_qvel = 0
            self.error_q_int_ant = 0
            self.error_q_ant = 0

        if True: #controller_type == 'inverse_dynamics_operational_space':
            self.x_ref = np.zeros((self.n_timesteps, 3))
            self.xvel_ref = np.zeros((self.n_timesteps, 3))
            self.xacc_ref = np.zeros((self.n_timesteps, 3))
            self.r_ref = np.zeros((self.n_timesteps, 4))
            self.rvel_ref = np.zeros((3, ))
            self.racc_ref = np.zeros((self.n_timesteps, 3))
            # self.error_r = np.zeros((self.n_timesteps, 3))
            self.error_r = np.zeros((3, ))
            self.error_rvel = np.zeros((3,))
            # self.error_racc = np.zeros((3,))
            # self.rvel_ref = np.zeros((self.n_timesteps, 3))
            # self.racc_ref = np.zeros((self.n_timesteps, 3))
            self.x_log = np.zeros((self.n_timesteps, 3))
            self.r_log = np.zeros((self.n_timesteps, 4))
            self.error_x_ant = 0
            self.robot_rtb = rtb.models.DH.LWR4()


        self.Kp = None
        self.Kd = None
        self.Ki = None
        self.Kp_rot = None
        self.Kd_rot = None
        self.J_ant = None
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
        n_timesteps = int((tf - ti) / dt)
        time = np.linspace(ti, tf, n_timesteps)
        k = int(ti / self.dt)

        self.q_ref[k:] = qd

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
            for idx, t in enumerate(time):
                i = idx + k - 1
                self.q_ref[i] = np.dot(np.array([1, (t - ti), (t - ti) ** 2, (t - ti) ** 3, (t - ti) ** 4, (t - ti) ** 5]),
                                  coeffs)
                self.qvel_ref[i] = np.dot(
                    np.array([0, 1, 2 * (t - ti), 3 * (t - ti) ** 2, 4 * (t - ti) ** 3, 5 * (t - ti) ** 4]), coeffs)
                self.qacc_ref[i] = np.dot(np.array([0, 0, 2, 6 * (t - ti), 12 * (t - ti) ** 2, 20 * (t - ti) ** 3]), coeffs)

    def trajectory_gen_operational_space(self, xd, xd_mat, tf, ti=0, dt=0.002, x_act=np.zeros((3,)), xmat_act=np.zeros((9,9)),
                                    traj=None):
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

        n_timesteps = int(tf / dt)
        time_spline = np.linspace(ti, tf+ti, n_timesteps)
        k = int(ti / self.dt)
        # self.r_ref[:] = self.get_euler_angles(mat=xd_mat)
        quat_ref = np.zeros(4)
        mujoco_py.functions.mju_mat2Quat(quat_ref, xd_mat.flatten())

        self.x_ref[k:] = xd
        self.r_ref[k:] = quat_ref

        if traj == 'spline3':
            T = tf
            x0 = x_act
            xvel0 = np.zeros((3,))
            xf = xd
            xvelf = np.zeros((3,))
            X = np.array([x0, xvel0, xf, xvelf])
            A_inv = np.linalg.inv(np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [1, T, T ** 2, T ** 3],
                                            [0, 1, 2 * T, 3 * T ** 2]]))
            coeffs = np.dot(A_inv, X)
            for i, t in enumerate(time_spline):
                self.x_ref[i] = np.dot(np.array([1, (t - ti), (t - ti) ** 2, (t - ti) ** 3]), coeffs)
                self.xvel_ref[i] = np.dot(np.array([0, 1, 2 * (t - ti), 3 * (t - ti) ** 2]), coeffs)
                self.xacc_ref[i] = np.dot(np.array([0, 0, 2, 6 * (t - ti)]), coeffs)

        if traj == 'spline5':
            T = tf
            x0 = x_act
            xvel0 = np.zeros((3,))
            xacc0 = np.zeros((3,))
            xf = xd
            xvelf = np.zeros((3,))
            xaccf = np.zeros((3,))
            X = np.array([x0, xvel0, xacc0, xf, xvelf, xaccf])
            A_inv = np.linalg.inv(np.array([[1, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0],
                                            [0, 0, 2, 0, 0, 0],
                                            [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
                                            [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                                            [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]]))
            coeffs = np.dot(A_inv, X)
            for i, t in enumerate(time_spline):
                self.x_ref[i] = np.dot(np.array([1, (t - ti), (t - ti) ** 2, (t - ti) ** 3, (t - ti) ** 4, (t - ti) ** 5]),
                                  coeffs)
                self.xvel_ref[i] = np.dot(
                    np.array([0, 1, 2 * (t - ti), 3 * (t - ti) ** 2, 4 * (t - ti) ** 3, 5 * (t - ti) ** 4]), coeffs)
                self.xacc_ref[i] = np.dot(np.array([0, 0, 2, 6 * (t - ti), 12 * (t - ti) ** 2, 20 * (t - ti) ** 3]), coeffs)
        print("end")

    def ctrl_independent_joints(self):
        error_q_int = (self.error_q + self.error_q_ant) * self.dt / 2 + self.error_q_int_ant
        self.error_q_int_ant = error_q_int
        return self.Kp.dot(self.error_q) + self.Kd.dot(self.error_qvel) + self.Ki.dot(error_q_int)

    def ctrl_inverse_dynamics_operational_space(self, sim, k):
        H = self.get_inertia_matrix(sim)
        H_inv = np.linalg.inv(H)
        C = self.get_coriolis_vector(sim)

        jacp, jacr = self.get_jacobian_site(sim)

        J = np.vstack((jacp, jacr))
        # T = np.array(np.bmat([[np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), sim.data.get_site_xmat(self.name_tcp)]]))
        # Jt = T.dot(J)
        # J = Jt

        if self.J_ant is None:
            self.J_ant = np.zeros(J.shape)
            J_dot = np.zeros(J.shape)
        else:
            J_dot = (J - self.J_ant) / self.dt

        J_inv = np.linalg.pinv(J)
        # Jt_inv = J_inv.T


        # equations from Robotics Handbook chapter 3, section 3.3
        H_op_space = np.linalg.inv(np.dot(J, np.dot(H_inv, J.T)))

        J_bar = H_inv.dot(J.T.dot(H_op_space))


        # NULL SPACE
        # xd_mat = np.array([[3.72030973e-01, -1.52734025e-03, 9.28219059e-01],
        #                    [-1.06081268e-03, -9.99998693e-01, -1.22027561e-03],
        #                    [9.28219710e-01, -5.30686220e-04, -3.72032107e-01]])
        # xd = np.array([9.92705091e-01, - 2.50066075e-04, 1.76208494e+00])
        projection_matrix_null_space = np.eye(7) - J_bar.dot(J)
        # T_fkine = np.array(np.bmat([[np.bmat([xd_mat, xd.reshape(3, 1)])], [np.bmat([np.bmat([np.zeros(3,), [1]])])]]))
        # Txd = smath.SE3(xd)
        # Txd.R[:] = xd_mat
        # qd = self.robot_rtb.ikine(Txd, q0=sim.data.qpos)
        tau_null_space = projection_matrix_null_space.dot(5*np.eye(7).dot(sim.data.qpos))


        # H_op_space = Jt_inv.dot(H.dot(J_inv))
        # C_op_space = np.dot(H_op_space, np.dot(J, np.dot(H_inv, C)) - np.dot(J_dot, sim.data.qvel))
        # C_op_space = np.linalg.pinv(J.T).dot(C.dot(J_inv.dot(J))) - H_op_space.dot(J.dot(J_inv.dot(J)))
        C_op_space = J_bar.T.dot(C) - H_op_space.dot(J_dot.dot(sim.data.qvel))

        # angle
        # quat_ref = np.zeros((4,))
        # quat_act = np.zeros((4,))
        # quat_vel_ref = np.zeros((3,))
        # quat_acc_ref = np.zeros((3,))

        # mujoco_py.functions.mju_mat2Quat(quat_ref, xd_mat.flatten())
        # mujoco_py.functions.mju_mat2Quat(quat_act, sim.data.get_site_xmat(self.name_tcp).flatten())

        # erro_quat = np.zeros((3,))
        # mujoco_py.functions.mju_subQuat(erro_quat, quat_ref, quat_act)
        # erro_vel_quat = quat_vel_ref
        # mujoco_py.functions.mju_subQuat(erro_quat, qa, qb)

        v_ = np.concatenate((self.xacc_ref[k], np.zeros(3, ))) +\
             np.dot(self.Kd, np.concatenate((self.error_xvel, self.error_rvel))) +\
             np.dot(self.Kp, np.concatenate((self.error_x, self.error_r)))


        # erro_euler = self.get_euler_from_quat(erro_quat)
        # r_ = erro_euler*0 + self.Kp.dot(erro_euler) + self.Kd.dot(erro_euler*0)
        # r_ = quat_ref + self.Kd.dot(self.)

        f = np.dot(H_op_space, v_) + C_op_space

        tau = np.dot(J.T, f)

        tau_max = 50

        if (np.absolute(tau) > tau_max).all():
            for i, tau_i in enumerate(tau):
                tau[i] = np.sign(tau_i) * tau_max

        return tau + tau_null_space

    def ctrl_inverse_dynamics(self, sim, k):
        H = self.get_inertia_matrix(sim)
        C = self.get_coriolis_vector(sim)

        v_ = self.qacc_ref[k] + np.dot(self.Kd, self.error_qvel) + np.dot(self.Kp, self.error_q)

        tau = np.dot(H, v_) + C

        return tau

    def ctrl_action(self, sim, k):
        if self.controller_type == 'independent_joints':
            if self.Kp is None:
                self.get_pd_matrices()
            u = self.ctrl_independent_joints()
            if self.use_gravity:
                u += self.tau_g(sim)

        if self.controller_type == 'inverse_dynamics':
            if self.Kp is None:
                self.get_pd_matrices()
            u = self.ctrl_inverse_dynamics(sim, k)

        if self.controller_type == 'inverse_dynamics_operational_space':
            if self.Kp is None:
                self.get_pd_matrices()
            u = self.ctrl_inverse_dynamics_operational_space(sim, k)


        return u

    def calculate_errors(self, sim_handle, k, qd_new=0, xd_new=0):
        if self.controller_type == 'inverse_dynamics_operational_space':
            xpos = sim_handle.data.get_site_xpos(self.name_tcp)
            xvel = sim_handle.data.get_site_xvelp(self.name_tcp)
            rpos = sim_handle.data.get_site_xmat(self.name_tcp)
            self.error_x = self.x_ref[k] + xd_new - xpos
            self.error_xvel = self.xvel_ref[k] - xvel
            # self.error_r = self.r_ref - rpos
            quat_ref = np.zeros((4,))
            mujoco_py.functions.mju_mat2Quat(quat_ref, rpos.flatten())
            mujoco_py.functions.mju_subQuat(self.error_r, self.r_ref[k], quat_ref)
            self.error_rvel = self.rvel_ref
        else:
            qpos = sim_handle.data.qpos
            qvel = sim_handle.data.qvel
            self.error_q = self.q_ref[k] + qd_new - qpos
            self.error_qvel = self.qvel_ref[k] - qvel

    def kuka_subtree_mass(self):
        body_names = ['kuka_link_{}'.format(i + 1) for i in range(7)]
        body_ids = [self.sim.model.body_name2id(n) for n in body_names]
        return self.sim.model.body_subtreemass[body_ids]

    def get_pd_matrices(self):

        if self.controller_type == 'inverse_dynamics_operational_space':
            n_space = 6
        else:
            n_space = 7

        subtree_mass = self.kuka_subtree_mass()
        Kp = np.eye(n_space)
        self.Kp_rot = np.eye(n_space+1)
        for i in range(n_space):
            if self.controller_type == 'inverse_dynamics_operational_space':
                Kp[i, i] = self.kp
                # self.Kp_rot[i,i] = self.kp
            else:
                Kp[i, i] = self.kp * subtree_mass[i]

        if self.use_kd:
            Kd = np.eye(n_space)
            self.Kd_rot = np.eye(n_space + 1)
            for i in range(n_space):
                if i == 6:
                    Kd[i, i] = Kp[i, i] ** (0.005 / self.kp)
                else:
                    Kd[i, i] = Kp[i, i] ** 0.5
                    # self.Kd_rot[i, i] = self.Kp_rot[i, i] ** 0.5
                    # for i in range(7):
            #     if i == 6:
            #         Kd[i, i] = Kp[i, i] ** 0.005 * (7 - i)
            #     elif i == 4:
            #         Kd[i, i] = Kp[i, i] ** 0.1 * (10 - i)
            #     else:
            #         Kd[i, i] = Kp[i, i] ** 0.25 * (10 - i)
        else:
            Kd = np.zeros((n_space, n_space))

        if self.use_ki:
            Ki = np.eye(n_space)
            for i in range(n_space):
                Ki[i, i] = Kp[i, i] * Kd[i, i] / self.lambda_H ** 2
        else:
            Ki = np.zeros((n_space, n_space))
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def tau_g(self, sim):
        Jp_shape = (3, sim.model.nv)
        comp = np.zeros((sim.model.nv,))
        for body, mass in zip(self.name_bodies, self.mass_links):
            Jp = sim.data.get_body_jacp(body).reshape(Jp_shape)
            comp = comp - np.dot(Jp.T, sim.model.opt.gravity * mass)
        return comp

    def plots(self):
        if self.controller_type == 'inverse_dynamics_operational_space':
            plt.plot(self.time_log, self.x_log)
            plt.plot(self.time_log, [x_r for x_r in self.x_ref], 'k--')
            plt.legend(['x' + str(i + 1) for i in range(3)])

            # plt.plot(self.time_log, self.r_log)
            # plt.plot(self.time_log, [quat_r for quat_r in self.r_ref], 'k--')
            # plt.legend(['quat' + str(i + 1) for i in range(4)])

            plt.show()
        else:
            plt.plot(self.time_log, self.q_log)
            plt.plot(self.time_log, [q_r for q_r in self.q_ref], 'k--')
            plt.legend(['q' + str(i + 1) for i in range(7)])
            plt.show()

    def step(self, sim, k):
        if self.controller_type == 'inverse_dynamics_operational_space':
            self.error_x_ant = self.error_x
            if self.plot_2d:
                self.x_log[k] = sim.data.get_site_xpos(self.name_tcp)
                rpos = sim.data.get_site_xmat(self.name_tcp)
                quat_log = np.zeros((4,))
                mujoco_py.functions.mju_mat2Quat(quat_log, rpos.flatten())
                self.r_log[k] = quat_log
                self.time_log[k] = sim.data.time
        else:
            self.error_q_ant = self.error_q
            if self.plot_2d:
                self.q_log[k] = sim.data.qpos
                self.time_log[k] = sim.data.time
        sim.step()

    def get_inertia_matrix(self, sim):
        # inertia matrix H
        mujoco_py.functions.mj_fullM(sim.model, self.H, sim.data.qM)
        H_ = self.H.reshape(sim.model.nv, sim.model.nv)
        return H_

    def get_coriolis_vector(self, sim):
        # internal forces: Coriolis + gravitational
        return sim.data.qfrc_bias

    def get_jacobian_site(self, sim):
        Jp_shape = (3, sim.model.nv)
        jacp  = sim.data.get_site_jacp(self.name_tcp).reshape(Jp_shape)
        jacr  = sim.data.get_site_jacr(self.name_tcp).reshape(Jp_shape)
        return jacp, jacr

    def get_euler_angles(self, sim=None, mat=None):
        if sim is not None:
            mat = sim.data.get_site_xmat(self.name_tcp)
        # notation according to springer handbook of robotics chap. 2, section 2
        beta = np.arctan2(-mat[2, 0], np.sqrt(mat[0,0]**2+mat[1,0]**2))
        alpha = np.arctan2(mat[1, 0]/np.cos(beta), mat[0,0]/np.cos(beta))
        gamma = np.arctan2(mat[2, 1]/np.cos(beta), mat[2,2]/np.cos(beta))
        return alpha, beta, gamma

    def get_euler_from_quat(self, quat):
        beta = np.arctan2(2*(quat[0]*quat[1]+quat[2]*quat[3]), 1-2*(quat[1]**2+quat[2]**2))
        alpha = np.arcsin(2*(quat[0]*quat[2]-quat[3]*quat[1]))
        gamma = np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]), 1-2*(quat[2]**2+quat[3]**2))
        return np.array([beta, alpha, gamma])

