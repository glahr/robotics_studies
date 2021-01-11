import numpy as np
import mujoco_py
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
import spatialmath as smath
from enum import Enum, auto

class TrajectoryProfile(Enum):
    SPLINE3 = auto()
    SPLINE5 = auto()
    STEP    = auto()

class CtrlType(Enum):
    INDEP_JOINTS = auto()
    INV_DYNAMICS = auto()
    INV_DYNAMICS_OP_SPACE = auto()


class CtrlUtils:

    def __init__(self, sim_handle, simulation_time, use_gravity, plot_2d, use_kd, use_ki, controller_type,
                 lambda_H=10.5, kp=20):
        self.sim = sim_handle
        self.dt = sim_handle.model.opt.timestep
        self.use_gravity = use_gravity
        self.plot_2d = plot_2d
        self.mass_links = self.sim.model.body_mass[4:11]
        self.name_bodies = [self.sim.model.body_id2name(i) for i in range(4, 11)]
        self.xpos_kuka_base = self.sim.data.get_body_xpos('kuka_base')
        self.name_tcp = self.sim.model.site_id2name(1)
        self.name_ft_sensor = self.sim.model.site_id2name(2)

        self.n_timesteps = int(simulation_time / self.dt)

        self.time_log = np.zeros((self.n_timesteps, 1))
        self.H = np.zeros(self.sim.model.nv * self.sim.model.nv)  # inertia matrix
        self.C = np.zeros((7,))  # Coriolis vector
        self.qd = np.zeros((self.sim.model.nv,))
        self.xd = np.zeros(3)

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
            self.last_qpos = np.zeros(sim_handle.model.nv)
            self.qpos_int = np.zeros(sim_handle.model.nv)

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
            self.q_nullspace = 0


        self.Kp = None
        self.Kd = None
        self.Ki = None
        self.Kp_rot = None
        self.Kd_rot = None
        self.J_ant = None
        self.use_kd = use_kd
        self.use_ki = use_ki
        self.lambda_H = lambda_H
        self.kp = kp

    def ctrl_independent_joints(self):
        error_q_int = (self.error_q + self.error_q_ant) * self.dt / 2 + self.error_q_int_ant
        self.error_q_int_ant = error_q_int
        return self.Kp.dot(self.error_q) + self.Kd.dot(self.error_qvel) + self.Ki.dot(error_q_int)

    def ctrl_inverse_dynamics_operational_space(self, sim, k, xacc_ref, alpha_ref):
        H = self.get_inertia_matrix(sim)
        H_inv = np.linalg.inv(H)
        C = self.get_coriolis_vector(sim)

        J = self.get_jacobian_site(sim)

        # J = np.vstack((jacp, jacr))
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

        # # Moore-penrose pseudoinverse  A^# = (A^TA)^-1 * A^T with A = J^T
        # JTpinv = np.linalg.inv(J * J.T) * J
        # lambda_ = np.linalg.inv(J * M_inv * J.T)
        #
        # # Null space projector
        # N = (eye(6) - J.T * JTpinv)
        # # null space torques (postural task)
        # tau0 = 50 * (conf.q0 - q) - 10 * qd
        # tau_null = N * tau0


        # NULL SPACE
        # xd_mat = np.array([[3.72030973e-01, -1.52734025e-03, 9.28219059e-01],
        #                    [-1.06081268e-03, -9.99998693e-01, -1.22027561e-03],
        #                    [9.28219710e-01, -5.30686220e-04, -3.72032107e-01]])
        # xd = np.array([9.92705091e-01, - 2.50066075e-04, 1.76208494e+00])
        # projection_matrix_null_space = np.eye(7) - J_bar.dot(J)
        projection_matrix_null_space = np.eye(7) - J.T.dot(np.linalg.pinv(J.T))
        # T_fkine = np.array(np.bmat([[np.bmat([xd_mat, xd.reshape(3, 1)])], [np.bmat([np.bmat([np.zeros(3,), [1]])])]]))
        # Txd = smath.SE3(xd)
        # Txd.R[:] = xd_mat
        # qd = self.robot_rtb.ikine(Txd, q0=sim.data.qpos)
        tau0 = 50*(self.q_nullspace - sim.data.qpos)*0 + 10*self.tau_g(sim)
        tau_null_space = projection_matrix_null_space.dot(tau0)


        # H_op_space = Jt_inv.dot(H.dot(J_inv))
        # C_op_space = np.dot(H_op_space, np.dot(J, np.dot(H_inv, C)) - np.dot(J_dot, sim.data.qvel))
        # C_op_space = (np.linalg.pinv(J.T).dot(C.dot(J_inv)) - H_op_space.dot(J.dot(J_inv))).dot(J.dot(sim.data.qvel))
        # C_op_space = np.linalg.pinv(J.T).dot(C.dot(J_inv)) - H_op_space.dot(J.dot(J_inv))

        # OK: Khatib
        C_op_space = J_bar.T.dot(C) - H_op_space.dot(J_dot.dot(sim.data.qvel))

        # OK: Handbook
        # C_op_space = H_op_space.dot(J.dot(H_inv.dot(C))-J_dot.dot(sim.data.qvel))

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

        v_ = np.concatenate((xacc_ref, alpha_ref)) +\
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

    def ctrl_inverse_dynamics(self, sim, qacc_ref):
        H = self.get_inertia_matrix(sim)
        C = self.get_coriolis_vector(sim)

        v_ = qacc_ref + np.dot(self.Kd, self.error_qvel) + np.dot(self.Kp, self.error_q)

        if self.use_ki:
            error_int = (self.error_q - self.error_q_ant)*self.dt/2 + self.error_q_int_ant
            v_ += self.Ki.dot(error_int)
            self.error_q_ant = self.error_q
            self.error_q_int_ant = error_int

        tau = np.dot(H, v_) + C

        return tau

    def ctrl_action(self, sim, k, qacc_ref=0, xacc_ref=0, alpha_ref=0):
        if self.controller_type == CtrlType.INDEP_JOINTS:
            if self.Kp is None:
                self.get_pd_matrices()
            u = self.ctrl_independent_joints()
            if self.use_gravity:
                u += self.tau_g(sim)

        if self.controller_type == CtrlType.INV_DYNAMICS:
            if self.Kp is None:
                self.get_pd_matrices()
            u = self.ctrl_inverse_dynamics(sim, qacc_ref)

        if self.controller_type == CtrlType.INV_DYNAMICS_OP_SPACE:
            if self.Kp is None:
                self.get_pd_matrices()
            u = self.ctrl_inverse_dynamics_operational_space(sim, k, xacc_ref=xacc_ref, alpha_ref=alpha_ref)

        return u

    def calculate_errors(self, sim, k,  qpos_ref=0, qvel_ref=0, kin=0):
        if self.controller_type == CtrlType.INV_DYNAMICS_OP_SPACE:
            x_ref, xvel_ref, xacc_ref, quat_ref, w_ref, alpha_ref = kin
            xpos = sim.data.get_site_xpos(self.name_tcp)
            xvel = sim.data.get_site_xvelp(self.name_tcp)
            # self.error_x = self.x_ref[k] - xpos
            self.error_x = x_ref[:3] - xpos
            self.error_xvel = xvel_ref[:3] - xvel
            quat_act = self.get_site_quat_from_mat(sim, self.name_tcp)
            mujoco_py.functions.mju_subQuat(self.error_r, quat_ref, quat_act)
            J = self.get_jacobian_site(sim)
            w_act = J.dot(sim.data.qvel)[3:]
            self.error_rvel = w_ref - w_act
        else:
            qpos = sim.data.qpos
            qvel = sim.data.qvel
            self.error_q = qpos_ref - qpos
            self.error_qvel = qvel_ref - qvel

    def kuka_subtree_mass(self):
        body_names = ['kuka_link_{}'.format(i + 1) for i in range(7)]
        body_ids = [self.sim.model.body_name2id(n) for n in body_names]
        return self.sim.model.body_subtreemass[body_ids]

    def get_pd_matrices(self):

        if self.controller_type == CtrlType.INV_DYNAMICS_OP_SPACE:
            n_space = 6
        else:
            n_space = 7

        subtree_mass = self.kuka_subtree_mass()
        Kp = np.eye(n_space)
        self.Kp_rot = np.eye(n_space+1)
        for i in range(n_space):
            if self.controller_type == CtrlType.INV_DYNAMICS_OP_SPACE:
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
        if self.controller_type == CtrlType.INV_DYNAMICS_OP_SPACE:
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
        if self.controller_type == CtrlType.INV_DYNAMICS_OP_SPACE:
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
        return np.vstack((jacp, jacr))
        # return jacp, jacr

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

    def get_site_quat_from_mat(self, sim, site_name):
        xmat = sim.data.get_site_xmat(site_name)
        xquat = np.zeros((4,))
        mujoco_py.functions.mju_mat2Quat(xquat, xmat.flatten())
        return xquat

    def inv_kinematics(self, sim):
        q_act = sim.data.qpos
        J = self.get_jacobian_site(sim)
        J_inv = J.T.dot(np.linalg.inv(J.dot(J.T)))
        v_tcp = np.concatenate((sim.data.get_site_xvelp(self.name_tcp), sim.data.get_site_xvelr(self.name_tcp)))
        q_next = q_act + J_inv.dot(v_tcp)*sim.model.opt.timestep
        return q_next

    def move_to_joint_pos(self, qd, sim, viewer=None):
        self.qd = qd
        trajectory = TrajectoryJoint(qd, ti=sim.data.time, q_act=sim.data.qpos, traj_profile=TrajectoryProfile.SPLINE3)
        # NAO EDITAR
        eps = 3 * np.pi / 180
        k = 1

        while True:
            # qd = np.array([0, 0, 3/2*np.pi/2, 0, 0, -np.pi/2, 0])
            # print("tolerancia " + str(sim.data.time))

            qpos_ref, qvel_ref, qacc_ref = trajectory.next()
            self.calculate_errors(sim, k, qpos_ref=qpos_ref, qvel_ref=qvel_ref)

            if (np.absolute(self.qd - sim.data.qpos) < eps).all():
                return

            u = self.ctrl_action(sim, k, qacc_ref=qacc_ref)  # , erro_q, erro_v, error_q_int_ant=error_q_int_ant)
            sim.data.ctrl[:] = u
            self.step(sim, k)
            # sim.step()
            if viewer is not None:
                viewer.render()
            k += 1
            if k >= self.n_timesteps:  # and os.getenv('TESTING') is not None:
                return

    def move_to_point(self, xd, xd_mat, sim, viewer=None):
        self.xd = xd
        x_act = sim.data.get_site_xpos(self.name_tcp)
        k = 0
        # xd[0] -= 0.2
        x_act_mat = sim.data.get_site_xmat(self.name_tcp)
        trajectory = TrajectoryOperational((xd, xd_mat), ti=sim.data.time, pose_act=(x_act, x_act_mat),
                                           traj_profile=TrajectoryProfile.SPLINE3)
        self.kp = 50
        self.get_pd_matrices()
        eps = 0.003
        while True:
            kinematics = trajectory.next()
            self.calculate_errors(sim, k, kin=kinematics)

            if (np.absolute(self.xd - sim.data.get_site_xpos(self.name_tcp)) < eps).all():
                return

            u = self.ctrl_action(sim, k, xacc_ref=kinematics[2],
                                 alpha_ref=kinematics[5])  # , erro_q, erro_v, error_q_int_ant=error_q_int_ant)
            sim.data.ctrl[:] = u
            # ctrl.q_nullspace = ctrl.inv_kinematics(sim)
            self.step(sim, k)
            # sim.step()
            if viewer is not None:
                viewer.render()
            k += 1
            if k >= self.n_timesteps:  # and os.getenv('TESTING') is not None:
                return



class TrajGen:

    def __init__(self):
        self.extra_points = 0
        self.iterator = None

    def next(self):
        assert(self.iterator is not None)
        return next(self.iterator)

class TrajectoryJoint(TrajGen):

    def __init__(self, qd, ti, t_duration=1, dt=0.002, q_act=np.zeros((7,)), traj_profile=None):
        super(TrajectoryJoint, self).__init__()
        self.iterator = self._traj_implementation(qd, ti, t_duration, dt, q_act, traj_profile)

    def _traj_implementation(self, qd, ti, t_duration=1, dt=0.002, q_act=np.zeros((7,)), traj_profile=None):
        """
        Joint trajectory generation with different methods.
        :param qd: joint space desired final point
        :param t_duration: total time of travel
        :param n: number of time steps
        :param ti: initial time
        :param dt: time step
        :param q_act: current joint position
        :param traj_profile: type of trajectory 'step', 'spline3', 'spline5', 'trapvel'
        :return:
        """
        n_timesteps = int((t_duration) / dt)
        time = np.linspace(ti, t_duration + ti, n_timesteps)
        # tf = ti + t_duration

        if traj_profile == TrajectoryProfile.SPLINE3:
            T = t_duration
            qvel0 = np.zeros((7,))
            qvelf = np.zeros((7,))
            Q = np.array([q_act, qvel0, qd, qvelf])
            A_inv = np.linalg.inv(np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [1, T, T ** 2, T ** 3],
                                            [0, 1, 2 * T, 3 * T ** 2]]))
            coeffs = np.dot(A_inv, Q)
            for t in time:
                q_ref = np.dot(np.array([1, (t - ti), (t - ti) ** 2, (t - ti) ** 3]), coeffs)
                qvel_ref = np.dot(np.array([0, 1, 2 * (t - ti), 3 * (t - ti) ** 2]), coeffs)
                qacc_ref = np.dot(np.array([0, 0, 2, 6 * (t - ti)]), coeffs)
                yield q_ref, qvel_ref, qacc_ref

        if traj_profile == TrajectoryProfile.SPLINE5:
            T = t_duration
            qvel0 = np.zeros((7,))
            qacc0 = np.zeros((7,))
            qvelf = np.zeros((7,))
            qaccf = np.zeros((7,))
            Q = np.array([q_act, qvel0, qacc0, qd, qvelf, qaccf])
            A_inv = np.linalg.inv(np.array([[1, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0],
                                            [0, 0, 2, 0, 0, 0],
                                            [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
                                            [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                                            [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]]))
            coeffs = np.dot(A_inv, Q)
            for t in time:
                q_ref = np.dot(np.array([1, (t - ti), (t - ti) ** 2, (t - ti) ** 3, (t - ti) ** 4, (t - ti) ** 5]),
                                  coeffs)
                qvel_ref = np.dot(
                    np.array([0, 1, 2 * (t - ti), 3 * (t - ti) ** 2, 4 * (t - ti) ** 3, 5 * (t - ti) ** 4]), coeffs)
                qacc_ref = np.dot(np.array([0, 0, 2, 6 * (t - ti), 12 * (t - ti) ** 2, 20 * (t - ti) ** 3]), coeffs)
                yield q_ref, qvel_ref, qacc_ref

        while True:
            self.extra_points = self.extra_points + 1
            yield q_ref, qvel_ref, qacc_ref

class TrajectoryOperational(TrajGen):

    def __init__(self, posed, ti, t_duration=1, dt=0.002, pose_act=np.zeros((7,)), traj_profile=None):
        super(TrajectoryOperational, self).__init__()
        self.iterator = self._traj_implementation(posed, ti, t_duration, dt, pose_act, traj_profile)

    def _traj_implementation(self, posed, ti, t_duration=1, dt=0.002, pose_act=np.zeros((7,)), traj_profile=None):
        """
        Joint trajectory generation with different methods.
        :param qd: joint space desired final point
        :param t_duration: total time of travel
        :param n: number of time steps
        :param ti: initial time
        :param dt: time step
        :param q_act: current joint position
        :param traj_profile: type of trajectory 'step', 'spline3', 'spline5', 'trapvel'
        :return:
        """

        xd = posed[0]
        xd_mat = posed[1]

        n_timesteps = int(t_duration / dt)
        time_spline = np.linspace(ti, t_duration + ti, n_timesteps)
        # k = int(ti / self.dt)
        # self.r_ref[:] = self.get_euler_angles(mat=xd_mat)
        quat_ref = np.zeros(4)
        mujoco_py.functions.mju_mat2Quat(quat_ref, xd_mat.flatten())
        quat_act = np.zeros(4)
        mujoco_py.functions.mju_mat2Quat(quat_act, pose_act[1].flatten())


        # self.x_ref[k:] = xd
        # self.r_ref[k:] = quat_ref

        if traj_profile == TrajectoryProfile.SPLINE3:
            T = t_duration
            x0 = pose_act[0]
            xvel0 = np.zeros((3,))
            xf = xd
            xvelf = np.zeros((3,))

            quat0 = quat_act
            w0 = np.zeros(4)
            quatf = quat_ref
            wf = np.zeros(4)


            X = np.array([x0, xvel0, xf, xvelf])
            W = np.array([quat0, w0, quatf, wf])
            A_inv = np.linalg.inv(np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [1, T, T ** 2, T ** 3],
                                            [0, 1, 2 * T, 3 * T ** 2]]))
            coeffsX = np.dot(A_inv, X)
            coeffsW = np.dot(A_inv, W)
            for i, t in enumerate(time_spline):
                x_ref = np.dot(np.array([1, (t - ti), (t - ti) ** 2, (t - ti) ** 3]), coeffsX)
                xvel_ref = np.dot(np.array([0, 1, 2 * (t - ti), 3 * (t - ti) ** 2]), coeffsX)
                xacc_ref = np.dot(np.array([0, 0, 2, 6 * (t - ti)]), coeffsX)

                quat_ref = np.dot(np.array([1, (t - ti), (t - ti) ** 2, (t - ti) ** 3]), coeffsW)
                w_ref = np.dot(np.array([0, 1, 2 * (t - ti), 3 * (t - ti) ** 2]), coeffsW)
                alpha_ref = np.dot(np.array([0, 0, 2, 6 * (t - ti)]), coeffsW)

                yield x_ref, xvel_ref, xacc_ref, quat_ref, w_ref[:3], alpha_ref[:3]


        # if traj_profile == TrajectoryProfile.SPLINE5:
        #     T = t_duration
        #     x0 = pose_act[:3]
        #     xvel0 = np.zeros((3,))
        #     xacc0 = np.zeros((3,))
        #     xf = xd
        #     xvelf = np.zeros((3,))
        #     xaccf = np.zeros((3,))
        #     X = np.array([x0, xvel0, xacc0, xf, xvelf, xaccf])
        #     A_inv = np.linalg.inv(np.array([[1, 0, 0, 0, 0, 0],
        #                                     [0, 1, 0, 0, 0, 0],
        #                                     [0, 0, 2, 0, 0, 0],
        #                                     [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
        #                                     [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
        #                                     [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]]))
        #     coeffs = np.dot(A_inv, X)
        #     for i, t in enumerate(time_spline):
        #         self.x_ref[i] = np.dot(np.array([1, (t - ti), (t - ti) ** 2, (t - ti) ** 3, (t - ti) ** 4, (t - ti) ** 5]),
        #                           coeffs)
        #         self.xvel_ref[i] = np.dot(
        #             np.array([0, 1, 2 * (t - ti), 3 * (t - ti) ** 2, 4 * (t - ti) ** 3, 5 * (t - ti) ** 4]), coeffs)
        #         self.xacc_ref[i] = np.dot(np.array([0, 0, 2, 6 * (t - ti), 12 * (t - ti) ** 2, 20 * (t - ti) ** 3]), coeffs)
        # print("end")
        while True:
            self.extra_points = self.extra_points + 1
            yield x_ref, xvel_ref, xacc_ref, quat_ref, w_ref[:3], alpha_ref[:3]  #TODO: returning first 3 values? study quaternion kinem.