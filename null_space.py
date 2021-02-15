import roboticstoolbox as rtb
import spatialmath as smath
import numpy as np
import scipy

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
        self.trajx = None
        self.q0 = None
        self.qf = None
        self.T0 = None
        self.Tf = None
        self.n = 500
        self.v = np.zeros(3)
        self.a = np.zeros(3)
        self.w = np.zeros(3)
        self.alpha = np.zeros(3)

    def ik_iiwa(self, xd, xdmat, q0=np.zeros(7)):
        Td = self.create_T(xd, xdmat)
        qd = self.iiwa.ikine(Td, q0=q0.reshape(1, 7), ilimit=200)
        return qd[0]

    def fk_iiwa(self, qd):
        fk = self.iiwa.fkine(qd)
        return fk.t, fk.R

    def traj_joint_generate(self, qd, q0, t=2, dt=0.002):
        self.k = 1
        self.n = int(t/dt)
        _, self.q, self.qvel, self.qacc = rtb.trajectory.jtraj(q0=q0, qf=qd, tv=np.linspace(0, t, self.n))

    def traj_joint_get_point(self):
        if self.k < self.n:
            self.k += 1
        return self.q[self.k - 1], self.qvel[self.k - 1], self.qacc[self.k - 1]

    def create_T(self, x, xmat):
        T = smath.SE3(x) * smath.SE3.OA(xmat[:, 1], xmat[:, 2])
        return T

    def traj_cart_generate(self, xd, xdmat, x0, x0mat, dt=0.002, tmax=2):
        self.T0 = self.create_T(x0, x0mat)
        self.Tf = self.create_T(xd, xdmat)
        self.n = int(tmax / dt)
        self.k = -1  # restart k for interpolation

    def traj_cart_get_point(self, dt=0.002):
        if self.k < self.n-1:
            self.k += 1
        T = self.Tf.interp(s=self.k/self.n, start=self.T0)
        quat_d = smath.quaternion.UnitQuaternion(T.R)
        x = T.t
        return x, self.v, self.a, quat_d, self.w, self.alpha

    def get_quat_from_mat(self, xmat):
        return smath.UnitQuaternion(xmat).A


def get_jacobian(iiwa, q):
    return iiwa.iiwa.jacobe(q)


def get_pseudo_jacobian(J):  # right pseudo-inverse
    return J.T.dot(np.linalg.inv(J.dot(J.T)))


def get_error(x_act, quat_act, xd, quat_d):
    error_x = xd - x_act
    error_r = (smath.UnitQuaternion(quat_d)-smath.UnitQuaternion(quat_act)).A[1:]
    # return np.concatenate((error_x, error_r))
    return error_x, error_r


def inverseKin(robot, q_init, q_nom, xpos_d, quat_d, reg=1e-4, upper=None, lower=None, cost_tol=1e-6, raise_on_fail=False, qpos_idx=None):
    '''
    Use SciPy's nonlinear least-squares method to compute the inverse kinematics
    '''

    if qpos_idx is None:
        qpos_idx = range(len(q_init))

    def residuals(q):
        robot.iiwa.q = q
        # xpos, xrot = forwardKin(sim, body_pos, identity_quat, body_id)
        xpos, xrot = robot.fk_iiwa(q)
        T = robot.create_T(xpos, xrot)
        quat = robot.get_quat_from_mat(T.R)
        error_x, error_r = get_error(xpos, quat, xpos_d, quat_d)
        res = np.concatenate((error_x, error_r, reg * (q - q_nom)))
        return res

    def jacobian(q):
        robot.iiwa.q = q
        J = get_jacobian(robot, q)
        residual_jacobian = np.vstack((J, reg*np.identity(q.size)))
        return residual_jacobian

    if lower is None:
        # lower = sim.model.jnt_range[qpos_idx,0]
        lower = robot.iiwa.qlim[0]
    else:
        lower = np.maximum(lower, robot.iiwa.qlim[0])

    if upper is None:
        # upper = sim.model.jnt_range[qpos_idx,1]
        lower = robot.iiwa.qlim[1]
    else:
        upper = np.minimum(upper, robot.iiwa.qlim[1])

    result = scipy.optimize.least_squares(residuals, q_init, bounds=(lower, upper))
    # result = scipy.optimize.least_squares(residuals, q_init, jac=jacobian, bounds=(lower, upper))

    if not result.success:
        print("Inverse kinematics failed with status: {}".format(result.status))
        if raise_on_fail:
            raise RuntimeError

    if result.cost > cost_tol:
        print("Inverse kinematics failed to find a sufficiently low cost")
        if raise_on_fail:
            raise RuntimeError("Infeasible")
    return result.x


if __name__ == '__main__':
    robot = KinematicsIiwa()

    # x0 = np.array([ 7.93423075e-01, -2.10110691e-04,  1.76392212e+00-1])
    # x0mat = np.array([[ 3.63644332e-01, -1.47347789e-03,  9.31536703e-01],
    #                 [-3.38816713e-04, -9.99998892e-01, -1.44950540e-03],
    #                 [ 9.31537807e-01,  2.11484219e-04, -3.63644428e-01]])
    q0 = np.array([0, 0.461, 0, -0.817, 0, 0.69, 0])
    robot.iiwa.q = q0

    # xd = np.array([4.01863413e-01, -4.95389989e-05, 1.53277615e+00-1])
    # xdmat = np.array([[0.99995499, -0.00210169, 0.00925189],
    #                   [-0.00209056, -0.99999708, -0.0012129],
    #                   [0.00925441, 0.0011935, -0.99995646]])

    T = robot.iiwa.fkine(q0+np.array([0, 0, 0, 0, 0, 0.4, 0]))
    xd = T.t
    # T = robot.create_T(xd, xdmat)
    quatd = robot.get_quat_from_mat(T.R)
    xveld = np.zeros(6)

    K = 10*np.eye(6)
    qvel0 = np.zeros(7)
    dt = 0.00002
    eps = 0.002
    eps_angle = np.pi/180
    e = np.ones(6)

    q_act = q0
    q_ant = q0
    qdot_ant = np.zeros(7)

    # pyplot = rtb.backends.PyPlot()
    # pyplot.add(robot.iiwa)

    alpha = 0.0001

    # while np.linalg.norm(e[:3]) > eps and np.linalg.norm(e[3:]) > eps_angle:
        # # try 1
        # x_act, x_actmat = robot.fk_iiwa(q_act)
        # T = robot.create_T(x_act, x_actmat)
        # quat_act = robot.get_quat_from_mat(T.R)
        #
        # e = get_error(x_act, quat_act, xd, quatd)
        #
        # J_A = get_jacobian(robot, q_act)
        # J_A_pseudo = get_pseudo_jacobian(J_A)
        # qdot = J_A_pseudo.dot(xveld + K.dot(e)) + (np.eye(7) - J_A_pseudo.dot(J_A)).dot(qvel0)
        # # qdot = J_A.T.dot(K.dot(e)) + (np.eye(7) - J_A_pseudo.dot(J_A)).dot(qvel0)
        #
        # q_act = q_ant + (qdot+qdot_ant)*dt/2
        # robot.iiwa.q = q_act
        # q_ant = q_act
        # qdot_ant = qdot
        # # print('interando!')
        # # print('q_act = ', q_act)
        # # print('erro = ', e)
        # # print('\n')
        # print('ex = ', e[:3], '\t\ter = ', e[3:])
        # # print(robot.iiwa.fkine(q_act))
        #
        # # pyplot.step()
        # # robot.iiwa.plot(q_act)
        
        # # try 2
        # x_act, x_actmat = robot.fk_iiwa(q_act)
        # T = robot.create_T(x_act, x_actmat)
        # quat_act = robot.get_quat_from_mat(T.R)
        # J = get_jacobian(robot, q_act)
        # J_pseudo = get_pseudo_jacobian(J)
        # e = get_error(x_act, quat_act, xd, quatd)
        # q_act = q_act + alpha*J_pseudo.dot(e)
        # print('ex = ', e[:3], '\t\ter = ', e[3:])

        # # try 3
    q_nom = np.zeros(7)
    world_pos = 0
    # robot, q_init, q_nom, xpos_d, quat_d
    q_opt = inverseKin(robot, q0, q_nom, xd, quatd)


    print('hi')