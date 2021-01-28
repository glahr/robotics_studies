import numpy as np
import mujoco_py
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
import spatialmath as smath
from enum import Enum, auto
from copy import deepcopy

class TrajectoryProfile(Enum):
    SPLINE3 = auto()
    SPLINE5 = auto()
    STEP    = auto()

class CtrlType(Enum):
    INDEP_JOINTS = auto()
    INV_DYNAMICS = auto()
    INV_DYNAMICS_OP_SPACE = auto()

class CtrlUtils:

    def __init__(self, sim_handle, simulation_time=10, plot_2d=False, use_kd=True, use_ki=True,
                 use_gravity=True, controller_type=CtrlType.INV_DYNAMICS, lambda_H=5, kp=20):
        self.sim = sim_handle
        self.dt = sim_handle.model.opt.timestep
        self.use_gravity = use_gravity
        self.plot_2d = plot_2d
        self.mass_links = self.sim.model.body_mass[4:11]
        self.name_bodies = [self.sim.model.body_id2name(i) for i in range(4, 11)]
        self.xpos_kuka_base = self.sim.data.get_body_xpos('kuka_base')
        self.name_tcp = self.sim.model.site_id2name(1)
        self.name_ft_sensor = self.sim.model.site_id2name(2)
        # self.name_hole_top = self.sim.model.site_id2name(3)
        # self.name_hole_base = self.sim.model.site_id2name(4)

        self.n_timesteps = int(simulation_time / self.dt)
        self.nv = 7

        self.time_log = np.zeros((self.n_timesteps, 1))
        self.H = np.zeros(self.nv * self.nv)  # inertia matrix
        self.C = np.zeros(self.nv)  # Coriolis vector
        self.qd = np.zeros(self.nv)
        self.xd = np.zeros(3)

        self.controller_type = controller_type

        # Definition of step vectors.
        if True: # controller_type == 'independent_joints' or controller_type == 'inverse_dynamics':
            self.q_ref = np.zeros(self.nv)
            self.qvel_ref = np.zeros(self.nv) #[np.zeros((7,)) for _ in range(self.n_timesteps)]
            self.qacc_ref = np.zeros(self.nv) #[np.zeros((7,)) for _ in range(self.n_timesteps)]
            self.q_log = np.zeros((self.n_timesteps, self.nv))
            self.error_int = np.zeros(self.nv)
            self.error_q = np.zeros(self.nv)
            self.error_qvel = np.zeros(self.nv)
            self.error_q_int_ant = np.zeros(self.nv)
            self.error_q_ant = np.zeros(self.nv)
            self.last_qpos = np.zeros(self.nv)
            self.qpos_int = np.zeros(self.nv)

        if True: #controller_type == 'inverse_dynamics_operational_space':
            self.x_ref = np.zeros((self.n_timesteps, 3))
            self.xvel_ref = np.zeros(3)
            self.xacc_ref = np.zeros(3)
            self.r_ref = np.zeros(4)
            self.rvel_ref = np.zeros(3)
            self.racc_ref = np.zeros(3)
            # self.error_r = np.zeros((self.n_timesteps, 3))
            self.error_r = np.zeros(3)
            self.error_rvel = np.zeros(3)
            # self.error_racc = np.zeros((3,))
            # self.rvel_ref = np.zeros((self.n_timesteps, 3))
            # self.racc_ref = np.zeros((self.n_timesteps, 3))
            self.x_log = np.zeros((self.n_timesteps, 3))
            self.r_log = np.zeros((self.n_timesteps, 4))
            self.error_x_ant = 0
            self.error_x_int = 0
            self.error_x_int_ant = 0
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

        self.iiwa_kin = KinematicsIiwa()

    def ctrl_independent_joints(self):
        error_q_int = (self.error_q + self.error_q_ant) * self.dt / 2 + self.error_q_int_ant
        self.error_q_int_ant = error_q_int
        return self.Kp.dot(self.error_q) + self.Kd.dot(self.error_qvel) + self.Ki.dot(error_q_int)

    def ctrl_inverse_dynamics_operational_space(self, sim, k, xacc_ref, alpha_ref):
        # H = self.get_inertia_matrix(sim)
        # H_inv = np.linalg.inv(H)
        # C = self.get_coriolis_vector(sim)
        J = self.get_jacobian_site(sim)
        #
        # if self.J_ant is None:
        #     self.J_ant = np.zeros(J.shape)
        #     J_dot = np.zeros(J.shape)
        # else:
        #     J_dot = (J - self.J_ant) / self.dt
        #     # self.J_ant = J
        #
        # # J_inv = np.linalg.pinv(J)
        # J_inv = J.T.dot(np.linalg.inv(J.dot(J.T)))
        # # Jt_inv = J_inv.T
        #
        # # equations from Robotics Handbook chapter 3, section 3.3
        # H_op_space = np.linalg.pinv(J.dot(H_inv.dot(J.T)))
        #
        # J_bar = H_inv.dot(J.T.dot(H_op_space))
        #
        # # # Moore-penrose pseudoinverse
        # # A^# = (A^TA)^-1 * A^T with A = J^T
        # # JTpinv = np.linalg.inv(J * J.T) * J
        # # lambda_ = np.linalg.inv(J * M_inv * J.T)
        # #
        # # # Null space projector
        # # N = (eye(6) - J.T * JTpinv)
        # # # null space torques (postural task)
        # # tau0 = 50 * (conf.q0 - q) - 10 * qd
        # # tau_null = N * tau0
        #
        #
        # NULL SPACE
        # projection_matrix_null_space = np.eye(7) - J_bar.dot(J)
        projection_matrix_null_space = np.eye(7) - J.T.dot(np.linalg.pinv(J.T))
        tau0 = 20*(self.q_nullspace - self.get_robot_qpos(sim)) + 50*self.tau_g(sim)*0
        tau_null_space = projection_matrix_null_space.dot(tau0)
        #
        # # H_op_space = Jt_inv.dot(H.dot(J_inv))
        # # C_op_space = (np.linalg.pinv(J.T).dot(C.dot(J_inv)) - H_op_space.dot(J.dot(J_inv))).dot(J.dot(self.get_robot_qvel(sim)))
        # # C_op_space = np.linalg.pinv(J.T).dot(C.dot(J_inv)) - H_op_space.dot(J.dot(J_inv))
        #
        # # OK: Khatib
        # # C_op_space = J_bar.T.dot(C) - H_op_space.dot(J_dot.dot(self.get_robot_qvel(sim)))
        #
        # # OK: Handbook
        # C_op_space = H_op_space.dot(J.dot(H_inv.dot(C))-J_dot.dot(self.get_robot_qvel(sim)))

        v_ = np.concatenate((xacc_ref, alpha_ref)) +\
             self.Kd.dot(np.concatenate((self.error_xvel, self.error_rvel))) +\
             self.Kp.dot(np.concatenate((self.error_x, self.error_r)))

        # TODO: implement integrative control action for operational space
        # if self.use_ki:

        sim.data.qacc[:] = v_

        mujoco_py.functions.mj_inverse(sim.model, sim.data)

        tau = sim.data.qfrc_inverse.copy()

        f = H_op_space.dot(v_) + C_op_space

        tau = J.T.dot(f)  # + C

        # joint torque limiting if needed
        # tau_max = 50
        # if (np.absolute(tau) > tau_max).all():
        #     for i, tau_i in enumerate(tau):
        #         tau[i] = np.sign(tau_i) * tau_max

        # new_tau = H.dot(J_inv.dot(v_) - J_dot.dot(self.get_robot_qvel(sim))) + C
        #
        return tau + tau_null_space
        # return new_tau

    def ctrl_inverse_dynamics(self, sim, qacc_ref):
        # H = self.get_inertia_matrix(sim)
        # C = self.get_coriolis_vector(sim)

        v_ = qacc_ref + self.Kd.dot(self.error_qvel) + self.Kp.dot(self.error_q)

        if self.use_ki:
            self.error_int = (self.error_q + self.error_q_ant)*self.dt/2 + self.error_q_int_ant
            v_ += self.Ki.dot(self.error_int)
            self.error_q_ant = self.error_q
            self.error_q_int_ant = self.error_int

        # tau = np.dot(H, v_) + C

        sim.data.qacc[:] = v_

        mujoco_py.functions.mj_inverse(sim.model, sim.data)

        tau = sim.data.qfrc_inverse.copy()

        return tau

    def _clear_integral_variables(self):
        if CtrlType.INV_DYNAMICS:
            self.error_int = np.zeros(7)
            self.error_q_int_ant = np.zeros(7)
            self.error_q_ant = np.zeros(7)
        if CtrlType.INV_DYNAMICS_OP_SPACE:
            self.error_int = np.zeros(7)
            self.error_q_int_ant = np.zeros(7)
            self.error_q_ant = np.zeros(7)

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
            J = self.get_jacobian_site(sim)
            w_act = J.dot(self.get_robot_qvel(sim))[3:]
            xvel_act = J.dot(self.get_robot_qvel(sim))[:3]
            xpos_act = sim.data.get_site_xpos(self.name_tcp)
            # xvel_act = sim.data.get_site_xvelp(self.name_tcp)
            self.error_x = x_ref[:3] - xpos_act
            self.error_xvel = xvel_ref[:3] - xvel_act
            quat_act = self.get_site_quat_from_mat(sim, self.name_tcp)
            # mujoco_py.functions.mju_subQuat(self.error_r, quat_ref, quat_act)
            self.error_r = self.get_quat_error(quat_act, quat_ref)
            # self.error_rvel = w_ref - w_act
            self.error_rvel = w_ref - w_act - 40*np.eye(3).dot(self.get_quat_error(quat_act, quat_ref))
        else:
            qpos = self.get_robot_qpos(sim)
            qvel = self.get_robot_qvel(sim)
            self.error_q = qpos_ref - qpos
            self.error_qvel = qvel_ref - qvel

    def kuka_subtree_mass(self):
        body_names = ['kuka_link_{}'.format(i + 1) for i in range(7)]
        body_ids = [self.sim.model.body_name2id(n) for n in body_names]
        return self.sim.model.body_subtreemass[body_ids]

    def get_pd_matrices(self):
        self._get_kp()
        self._get_kd()
        self._get_ki()

    def _get_kp(self):
        if self.controller_type == CtrlType.INV_DYNAMICS_OP_SPACE:
            n_space = 6
        else:
            n_space = 7
        subtree_mass = self.kuka_subtree_mass()
        Kp = np.eye(n_space)
        self.Kp_rot = np.eye(n_space + 1)
        for i in range(n_space):
            if self.controller_type == CtrlType.INV_DYNAMICS_OP_SPACE:
                if i < 3:  # xyz
                    Kp[i, i] = self.kp
                else:  # abc
                    Kp[i, i] = self.kp * 1.5
                # self.Kp_rot[i,i] = self.kp
            else:
                Kp[i, i] = self.kp * subtree_mass[i]
        self.Kp = Kp

    def _get_kd(self):
        if self.controller_type == CtrlType.INV_DYNAMICS_OP_SPACE:
            n_space = 6
        else:
            n_space = 7

        if self.use_kd:
            Kd = np.eye(n_space)
            # self.Kd_rot = np.eye(n_space + 1)
            for i in range(n_space):
                if i == 6 and self.controller_type != CtrlType.INV_DYNAMICS_OP_SPACE:
                    Kd[i, i] = self.Kp[i, i] ** (0.005 / self.kp)
                else:
                    Kd[i, i] = self.Kp[i, i] ** 0.5
                    # self.Kd_rot[i, i] = self.Kp_rot[i, i] ** 0.5
                    # for i in range(7):
        else:
            Kd = np.zeros((n_space, n_space))

        self.Kd = Kd

    def _get_ki(self):
        if self.controller_type == CtrlType.INV_DYNAMICS_OP_SPACE:
            n_space = 6
        else:
            n_space = 7

        if self.use_ki:
            Ki = np.eye(n_space)
            for i in range(n_space):
                Ki[i, i] = self.Kp[i, i] * self.Kd[i, i] / self.lambda_H ** 2
                if i == 6:
                    Ki[i, i] = self.Kp[i, i] * self.Kd[i, i] / self.lambda_H
        else:
            Ki = np.zeros((n_space, n_space))

        self.Ki = Ki

    def tau_g(self, sim):
        Jp_shape = (3, self.nv)
        comp = np.zeros((self.nv,))
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
                self.q_log[k] = self.get_robot_qpos(sim)
                self.time_log[k] = sim.data.time
        sim.step()

    def get_inertia_matrix(self, sim):
        # inertia matrix H
        mujoco_py.functions.mj_fullM(sim.model, self.H, sim.data.qM)
        return self.H.reshape(sim.model.nv, sim.model.nv)[:self.nv, :self.nv]

    def get_coriolis_vector(self, sim):
        # internal forces: Coriolis + gravitational
        return sim.data.qfrc_bias[:self.nv]

    def get_jacobian_site(self, sim):
        Jp_shape = (3, sim.model.nv)
        jacp  = sim.data.get_site_jacp(self.name_tcp).reshape(Jp_shape)
        jacr  = sim.data.get_site_jacr(self.name_tcp).reshape(Jp_shape)
        return np.vstack((jacp, jacr))

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

    # def inv_kinematics(self, sim):
    #     q_act = self.get_robot_qpos(sim)
    #     J = self.get_jacobian_site(sim)
    #     J_inv = J.T.dot(np.linalg.inv(J.dot(J.T)))
    #     v_tcp = np.concatenate((sim.data.get_site_xvelp(self.name_tcp), sim.data.get_site_xvelr(self.name_tcp)))
    #     q_next = q_act + J_inv.dot(v_tcp)*sim.model.opt.timestep
    #     return q_next

    def move_to_joint_pos(self, sim, xd=None, xdmat=None, qd=None, viewer=None, eps=1.5*np.pi/180):
        if qd is not None:
            self.qd = qd
        if xd is not None:
            self.qd = self.iiwa_kin.ik_iiwa(xd - sim.data.get_body_xpos('kuka_base'), xdmat, q0=self.get_robot_qpos(sim))
        self._clear_integral_variables()
        # trajectory = TrajectoryJoint(self.qd, ti=sim.data.time, q0=self.get_robot_qpos(sim), traj_profile=TrajectoryProfile.SPLINE3)
        self.iiwa_kin.traj_joint_generate(self.qd, np.asarray(self.get_robot_qpos(sim)))

        k = 1

        while True:
            # qpos_ref, qvel_ref, qacc_ref = trajectory.next()
            qpos_ref, qvel_ref, qacc_ref = self.iiwa_kin.traj_joint_get_point()
            self.calculate_errors(sim, k, qpos_ref=qpos_ref, qvel_ref=qvel_ref)

            if (np.absolute(self.qd - self.get_robot_qpos(sim)) < eps).all():
                return

            u = self.ctrl_action(sim, k, qacc_ref=qacc_ref)
            sim.data.ctrl[:] = u
            self.step(sim, k)
            if viewer is not None:
                if qd is not None:
                    xd, xdmat = self.iiwa_kin.fk_iiwa(qd)
                    xd += sim.data.get_body_xpos('kuka_base')
                self.render_frame(viewer, xd, xdmat)
                viewer.render()
            k += 1
            # TODO: create other stopping criteria
            if k >= self.n_timesteps:  # and os.getenv('TESTING') is not None:
                return

    def move_to_point(self, xd, xdmat, sim, viewer=None):
        self.xd = xd
        x_act, x_act_mat = self.get_site_pose(sim)
        # x_act = sim.data.get_site_xpos(self.name_tcp)
        k = 0
        # x_act_mat = sim.data.get_site_xmat(self.name_tcp)
        trajectory = TrajectoryOperational((xd, xdmat), ti=sim.data.time,
                                           pose0=(x_act, x_act_mat), traj_profile=TrajectoryProfile.SPLINE3)
        # self.iiwa_kin.traj_cart_generate(xd, xdmat, x_act, x_act_mat)
        self.kp = 500
        self.get_pd_matrices()
        eps = 0.003

        self.q_nullspace = self.iiwa_kin.ik_iiwa(xd - sim.data.get_body_xpos('kuka_base'), xdmat, q0=self.get_robot_qpos(sim))[0]

        while True:
            kinematics = trajectory.next()
            # kinematics =
            self.calculate_errors(sim, k, kin=kinematics)

            # TODO: implement orientation error stopping criteria
            if (np.absolute(self.xd - sim.data.get_site_xpos(self.name_tcp)) < eps).all():
                return

            u = self.ctrl_action(sim, k, xacc_ref=kinematics[2],
                                 alpha_ref=kinematics[5])
            sim.data.ctrl[:] = u
            self.step(sim, k)
            if viewer is not None:
                self.render_frame(viewer, xd, xdmat)
                viewer.render()
            k += 1
            if k >= self.n_timesteps:  # and os.getenv('TESTING') is not None:
                return

    def get_robot_qpos(self, sim):
        return sim.data.qpos[0:7]

    def get_robot_qvel(self, sim):
        return sim.data.qvel[0:7]

    def render_frame(self, viewer, pos, mat):
        viewer.add_marker(pos=pos,
                          label='',
                          type=mujoco_py.generated.const.GEOM_SPHERE,
                          size=[.01, .01, .01])
        cylinder_half_height = 0.02
        pos_cylinder = pos + mat.dot([0.0, 0.0, cylinder_half_height])
        viewer.add_marker(pos=pos_cylinder,
                          label='',
                          type=mujoco_py.generated.const.GEOM_CYLINDER,
                          size=[.005, .005, cylinder_half_height],
                          mat=mat)

    def quat2Mat(self, quat):
        '''
        Convenience function for mju_quat2Mat.
        '''
        res = np.zeros(9)
        mujoco_py.functions.mju_quat2Mat(res, quat)
        res = res.reshape(3, 3)
        return res

    def get_site_pose(self, sim):
        xd = deepcopy(sim.data.get_site_xpos(self.name_tcp))
        xmat = deepcopy(sim.data.get_site_xmat(self.name_tcp))
        return xd, xmat

    def get_quat_error(self, quat_act, quat_ref):
        q = quat_act
        qd = quat_ref
        # q_error = np.concatenate((eta ,eps))
        q_error = qd[0] * q[1:] - q[0] * qd[1:] + smath.base.skew(qd[1:]).dot(q[1:])
        return q_error


class TrajGen:

    def __init__(self):
        self.extra_points = 0
        self.iterator = None

    def next(self):
        assert(self.iterator is not None)
        return next(self.iterator)


class TrajectoryJoint(TrajGen):

    def __init__(self, qd, ti, t_duration=1, dt=0.002, q0=np.zeros((7,)), traj_profile=None):
        super(TrajectoryJoint, self).__init__()
        self.q0 = q0
        self.qd = qd
        self.ti = ti
        self.n_timesteps = int((t_duration) / dt)
        self.time = np.linspace(ti, t_duration + ti, self.n_timesteps)
        self.iterator = self._traj_implementation(qd, ti, t_duration, dt, q0, traj_profile)

    def _traj_implementation(self, qd, ti, t_duration=1, dt=0.002, q0=np.zeros((7,)), traj_profile=None):
        """
        Joint trajectory generation with different methods.
        :param qd: joint space desired final point
        :param t_duration: total time of travel
        :param n: number of time steps
        :param ti: initial time
        :param dt: time step
        :param q0: current joint position
        :param traj_profile: type of trajectory 'step', 'spline3', 'spline5', 'trapvel'
        :return:
        """

        if traj_profile == TrajectoryProfile.SPLINE3:
            T = t_duration
            qvel0 = np.zeros((7,))
            qvelf = np.zeros((7,))
            Q = np.array([q0, qvel0, qd, qvelf])
            A_inv = np.linalg.inv(np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [1, T, T ** 2, T ** 3],
                                            [0, 1, 2 * T, 3 * T ** 2]]))
            coeffs = np.dot(A_inv, Q)
            for t in self.time:
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
            Q = np.array([q0, qvel0, qacc0, qd, qvelf, qaccf])
            A_inv = np.linalg.inv(np.array([[1, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0],
                                            [0, 0, 2, 0, 0, 0],
                                            [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
                                            [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                                            [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]]))
            coeffs = np.dot(A_inv, Q)
            for t in self.time:
                q_ref = np.dot(np.array([1, (t - ti), (t - ti) ** 2, (t - ti) ** 3, (t - ti) ** 4, (t - ti) ** 5]),
                                  coeffs)
                qvel_ref = np.dot(
                    np.array([0, 1, 2 * (t - ti), 3 * (t - ti) ** 2, 4 * (t - ti) ** 3, 5 * (t - ti) ** 4]), coeffs)
                qacc_ref = np.dot(np.array([0, 0, 2, 6 * (t - ti), 12 * (t - ti) ** 2, 20 * (t - ti) ** 3]), coeffs)
                yield q_ref, qvel_ref, qacc_ref

        while True:
            self.extra_points = self.extra_points + 1
            yield q_ref, qvel_ref, qacc_ref

    def plot_traj(self):
        q = []
        qtraj = self._traj_implementation(self.qd, self.ti, q0=self.q0, traj_profile=TrajectoryProfile.SPLINE3)
        print(qtraj)
        for i, t in enumerate(self.time):
            qpos, _, _ = qtraj.next()
            q.apend(qpos)
        plt.plot(self.time, qpos)


class TrajectoryOperational(TrajGen):

    def __init__(self, posed, ti, t_duration=1, dt=0.002, pose0=(np.zeros(3), np.eye(3)), traj_profile=None):
        super(TrajectoryOperational, self).__init__()
        self.iterator = self._traj_implementation(posed, ti, t_duration, dt, pose0, traj_profile)

    def _traj_implementation(self, posed, ti, t_duration=1, dt=0.002, pose0=(np.zeros(3), np.eye(3)), traj_profile=None):
        """
        Joint trajectory generation with different methods.
        :param posed: desired pose
        :param ti: initial time
        :param t_duration: total time of travel
        :param n: number of time steps
        :param ti: initial time
        :param dt: time step
        :param pose0: current pose
        :param traj_profile: type of trajectory 'step', 'spline3', 'spline5', 'trapvel'
        :return:
        """

        xf = posed[0]
        xf_mat = posed[1]
        x0 = pose0[0]
        xmat_act = pose0[1]

        n_timesteps = int(t_duration / dt)
        time_spline = np.linspace(ti, t_duration + ti, n_timesteps)

        quatf = np.zeros(4)
        mujoco_py.functions.mju_mat2Quat(quatf, xf_mat.flatten())
        quat0 = np.zeros(4)
        mujoco_py.functions.mju_mat2Quat(quat0, xmat_act.flatten())

        if traj_profile == TrajectoryProfile.SPLINE3:
            T = t_duration
            xvel0 = np.zeros(3)
            xvelf = np.zeros(3)

            quat0_v = np.zeros(4)
            quatf_v = np.zeros(4)
            w_ref = np.zeros(3)
            w_ant = np.zeros(3)
            quat_ant = deepcopy(quat0)

            A_inv = np.linalg.inv(np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [1, T, T ** 2, T ** 3],
                                            [0, 1, 2 * T, 3 * T ** 2]]))

            # Cartesian position constraints
            X = np.array([x0, xvel0, xf, xvelf])

            # Quaternions interp
            Q = np.array([quat0, quat0_v, quatf, quatf_v])

            coeffsX = np.dot(A_inv, X)
            coeffsQ = np.dot(A_inv, Q)
            for i, t in enumerate(time_spline):
                x_ref = np.dot(np.array([1, (t - ti), (t - ti) ** 2, (t - ti) ** 3]), coeffsX)
                xvel_ref = np.dot(np.array([0, 1, 2 * (t - ti), 3 * (t - ti) ** 2]), coeffsX)
                xacc_ref = np.dot(np.array([0, 0, 2, 6 * (t - ti)]), coeffsX)

                quat_ref = np.dot(np.array([1, (t - ti), (t - ti) ** 2, (t - ti) ** 3]), coeffsQ)
                mujoco_py.functions.mju_subQuat(w_ref, quat0, quat_ant)
                quat_ant = quat0
                alpha_ref = (w_ref - w_ant)/dt
                w_ant = w_ref

                yield x_ref, xvel_ref, xacc_ref, quatf, np.zeros(3), np.zeros(3)  # quat_ref, w_ref, alpha_ref

        while True:
            self.extra_points = self.extra_points + 1
            yield x_ref, xvel_ref, xacc_ref, quat_ref, w_ref, alpha_ref


class KinematicsIiwa:
    def __init__(self):
        # TODO: check all corrections, there are some mistakes, even though small
        self.iiwa = rtb.models.DH.LWR4()
        # correcting DH params
        self.iiwa.links[2].d = 0.42
        self.iiwa.links[4].d = 0.4
        # correcting tool params
        self.iiwa.tool.R[:] = np.eye(3)  # TODO: change tool length
        self.iiwa.tool.t[2] = 0.159 + 0.126 - 0.04498
        # correcting base params
        T_new = self.iiwa.base
        T_new.t[2] = 0.36
        T_new.R[:] = smath.SO3.Rz(np.pi) * T_new.R
        self.iiwa.base = T_new

        self.k = 0
        self.q = None
        self.qvel = None
        self.qacc = None
        self.traj = None
        self.n = 500

    def ik_iiwa(self, xd, xdmat, q0=np.zeros(7)):
        Td = self.create_T(xd, xdmat)
        qd = self.iiwa.ikine(Td, q0=q0.reshape(1, 7), ilimit=200)
        return qd[0]

    def fk_iiwa(self, qd):
        fk = self.iiwa.fkine(qd)
        return fk.t, fk.R

    def traj_joint_generate(self, qd, q0, t=2):
        self.k = 0
        _, self.q, self.qvel, self.qacc = rtb.trajectory.jtraj(q0=q0, qf=qd, tv=np.linspace(0, t, self.n))

    def traj_joint_get_point(self):
        if self.k < self.n:
            self.k += 1
        return self.q[self.k - 1], self.qvel[self.k - 1], self.qacc[self.k - 1]

    def create_T(self, x, xmat):
        T = smath.SE3(x) * smath.SE3.OA(xmat[:, 1], xmat[:, 2])
        return T

    def traj_cart_generate(self, xd, xdmat, x0, x0mat, dt=0.002, tmax=2):
        T0 = self.create_T(x0, x0mat)
        Tf = self.create_T(xd, xdmat)
        self.traj = rtb.trajectory.ctraj(T0, Tf, int(tmax/dt))

    def traj_cart_get_point(self, dt=0.002):
        x = self.traj[self.k].t
        if self.k > 1:
            v = (self.traj[self.k].t - self.traj[self.k-1].t)/dt
        else:
            v = np.zeros(3)
        if self.k > 2:
            a = (self.traj[self.k].t - 2*self.traj[self.k-1].t + self.traj[self.k-2].t)/dt
        else:
            a = np.zeros(3)



        if self.k < self.n:
            self.k += 1
        return


    # TODO: implement ctraj()