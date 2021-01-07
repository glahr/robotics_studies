#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
# from gym_kuka_mujoco.utils.kinematics import forwardKin, forwardKinJacobian, forwardKinSite, forwardKinJacobianSite
import mujoco_py
import numpy as np
import copy
from controllers_utils import CtrlUtils, TrajectoryOperational, TrajectoryJoint, TrajectoryProfile, CtrlType

def load_model_mujoco(simulate, use_gravity):
    model_name = "assets/full_kuka_all_joints"
    if use_gravity:
        model_name += "_gravity"
    model_xml = load_model_from_path(model_name + '.xml')
    sim = MjSim(model_xml)
    if simulate:
        viewer = MjViewer(sim)
    return sim, viewer


if __name__ == '__main__':

    simulate = True
    use_gravity = True
    plot_2d = True
    use_kd = True
    use_ki = True
    simulation_time = 10  # s

    qd = np.array([0, 0.461, 0, -0.817, 0, 0.69, 0])
    # xd = np.array([0.1, 0, 1.7])
    xd = np.array([9.92705091e-01, - 2.50066075e-04,  1.76208494e+00])
    xd_mat = np.array([[3.72030973e-01, -1.52734025e-03,  9.28219059e-01],
                       [-1.06081268e-03, -9.99998693e-01, -1.22027561e-03],
                       [9.28219710e-01, -5.30686220e-04, -3.72032107e-01]])
    # rd =
    # qd = np.array([0, 0, 0, 0, 0, 0, 0])
    qd2 = np.array([0, 0, 0, -np.pi/2, -np.pi/2, 0, 0])

    # controller_type = CtrlType.INDEP_JOINTS
    controller_type = CtrlType.INV_DYNAMICS
    # controller_type = CtrlType.INV_DYNAMICS_OP_SPACE

    sim, viewer = load_model_mujoco(simulate, use_gravity)
    sim.forward()
    ctrl = CtrlUtils(sim, simulation_time, use_gravity, plot_2d, use_kd, use_ki, controller_type)
    trajectory = TrajectoryJoint(qd, ti=sim.data.time, q_act=sim.data.qpos, traj_profile=TrajectoryProfile.SPLINE3)
    # trajectory.__next__()
    # trajectory.__str__()
    # str(trajectory)
    ctrl.kp = 20
    ctrl.qd = qd
    # ctrl.kp = 11
    ctrl.lambda_H = 3.5
    tf = 2  # spline final time

    # NAO EDITAR
    eps = 5*np.pi/180
    k = 1

    # TODO: just to leave singularity position
    while True:

        if (np.absolute(ctrl.error_q) < eps).all() and sim.data.time > 1:
            break
            # qd = np.array([0, 0, 3/2*np.pi/2, 0, 0, -np.pi/2, 0])
        # print("tolerancia " + str(sim.data.time))

        qpos_ref, qvel_ref, qacc_ref = trajectory.next()
        ctrl.calculate_errors(sim, k, qpos_ref=qpos_ref, qvel_ref=qvel_ref)
        u = ctrl.ctrl_action(sim, k, qacc_ref=qacc_ref)  # , erro_q, erro_v, error_q_int_ant=error_q_int_ant)
        sim.data.ctrl[:] = u
        ctrl.step(sim, k)
        # sim.step()
        if simulate:
            viewer.render()
        k += 1
        if k >= ctrl.n_timesteps:  # and os.getenv('TESTING') is not None:
            break

    # TODO: Operational space control
    k = 0
    ctrl.controller_type = CtrlType.INV_DYNAMICS_OP_SPACE
    xd = sim.data.get_site_xpos(ctrl.name_tcp)
    xd_mat = sim.data.get_site_xmat(ctrl.name_tcp)
    trajectory2 = TrajectoryOperational((xd, xd_mat), ti=sim.data.time, pose_act=(xd, xd_mat), traj_profile=TrajectoryProfile.SPLINE3)
    ctrl.kp = 50
    ctrl.get_pd_matrices()
    ctrl.q_nullspace = copy.deepcopy(sim.data.qpos)
    while True:
        kinematics = trajectory2.next()
        ctrl.calculate_errors(sim, k, kin=kinematics)
        u = ctrl.ctrl_action(sim, k, xacc_ref=kinematics[2], alpha_ref=kinematics[5])  # , erro_q, erro_v, error_q_int_ant=error_q_int_ant)
        sim.data.ctrl[:] = u
        ctrl.step(sim, k)
        # sim.step()
        if simulate:
            viewer.render()
        k += 1
        if k >= ctrl.n_timesteps:  # and os.getenv('TESTING') is not None:
            break

    if plot_2d:
        ctrl.plots()

    print(trajectory.extra_points)
    # print("biggest element = ", max_diag_element)
