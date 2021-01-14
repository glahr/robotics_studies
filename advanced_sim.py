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
    # xd = np.array([7.91954831e-01, -1.56225542e-05,  1.78926928e+00])
    # xd_mat = np.array([[ 3.10728547e-01, -8.31370781e-04,  9.50498332e-01],
    #                    [-2.43614713e-03, -9.99997030e-01, -7.82619487e-05],
    #                    [ 9.50495573e-01, -2.29123556e-03, -3.10729650e-01]])

    #TODO: take type definition from this stage

    # controller_type = CtrlType.INDEP_JOINTS
    controller_type = CtrlType.INV_DYNAMICS
    # controller_type = CtrlType.INV_DYNAMICS_OP_SPACE

    sim, viewer = load_model_mujoco(simulate, use_gravity)
    sim.step()
    ctrl = CtrlUtils(sim, simulation_time, use_gravity, plot_2d, use_kd, use_ki, controller_type)
    # trajectory = TrajectoryJoint(qd, ti=sim.data.time, q_act=sim.data.qpos, traj_profile=TrajectoryProfile.SPLINE3)
    # trajectory.__next__()
    # trajectory.__str__()
    # str(trajectory)
    # ctrl.kp = 20
    # ctrl.qd = qd
    # ctrl.kp = 11
    # ctrl.lambda_H = 3.5
    # tf = 2  # spline final time


    # qd = np.array([ 5.40407988e-19,  4.61000000e-01, -1.05586330e-18, -8.17000000e-01, 8.72502239e-19,  6.90000000e-01, -3.96547266e-19])
    qd = np.array([0, 0, 0, -np.pi / 2, 0, np.pi/2, 0])
    ctrl.move_to_joint_pos(sim, qd=qd, viewer=viewer)

    xd, xdmat = ctrl.get_site_pose(sim)

    xd[0] -= 0.05
    ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)
    xd[1] += 0.05
    ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)
    xd[0] += 0.05
    ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)
    xd[1] -= 0.05
    ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)

    # qd = np.array([0, -0.4, 0, -0.3, .5, 0.69, 0])
    # ctrl.move_to_joint_pos(qd, sim, viewer=viewer)
    #
    # qd = np.array([0, 0.1, 0, -0.6, 0, 0.13, 0])
    # ctrl.move_to_joint_pos(qd, sim, viewer=viewer)
    #
    # qd = np.array([np.pi/3, 0, 0, -np.pi/2, 0, 0, 0])
    # ctrl.move_to_joint_pos(qd, sim, viewer=viewer)
    #
    # q0 = np.zeros(7)
    # ctrl.move_to_joint_pos(q0, sim, viewer=viewer)

    # while True:
    #
    #     if (np.absolute(ctrl.error_q) < eps).all() and sim.data.time > 1:
    #         break
    #         # qd = np.array([0, 0, 3/2*np.pi/2, 0, 0, -np.pi/2, 0])
    #     # print("tolerancia " + str(sim.data.time))
    #
    #     qpos_ref, qvel_ref, qacc_ref = trajectory.next()
    #     ctrl.calculate_errors(sim, k, qpos_ref=qpos_ref, qvel_ref=qvel_ref)
    #     u = ctrl.ctrl_action(sim, k, qacc_ref=qacc_ref)  # , erro_q, erro_v, error_q_int_ant=error_q_int_ant)
    #     sim.data.ctrl[:] = u
    #     ctrl.step(sim, k)
    #     # sim.step()
    #     if simulate:
    #         viewer.render()
    #     k += 1
    #     if k >= ctrl.n_timesteps:  # and os.getenv('TESTING') is not None:
    #         break

    # TODO: Operational space control

    ctrl.controller_type = CtrlType.INV_DYNAMICS_OP_SPACE

    xd, xdmat = ctrl.get_site_pose(sim)

    xd[0] -= 0.05
    ctrl.move_to_point(xd, xdmat, sim, viewer=viewer)
    xd[1] += 0.05
    ctrl.move_to_point(xd, xdmat, sim, viewer=viewer)
    xd[0] += 0.05
    ctrl.move_to_point(xd, xdmat, sim, viewer=viewer)
    xd[1] -= 0.05
    ctrl.move_to_point(xd, xdmat, sim, viewer=viewer)


    # xd = sim.data.get_site_xpos(ctrl.name_tcp)
    # # xd[0] -= 0.2
    # xd_mat = sim.data.get_site_xmat(ctrl.name_tcp)
    # trajectory = TrajectoryOperational((xd, xd_mat), ti=sim.data.time, pose_act=(xd, xd_mat), traj_profile=TrajectoryProfile.SPLINE3)
    # ctrl.kp = 50
    # ctrl.get_pd_matrices()
    # ctrl.q_nullspace = copy.deepcopy(sim.data.qpos)
    # while True:
    #     kinematics = trajectory.next()
    #     ctrl.calculate_errors(sim, k, kin=kinematics)
    #     u = ctrl.ctrl_action(sim, k, xacc_ref=kinematics[2], alpha_ref=kinematics[5])  # , erro_q, erro_v, error_q_int_ant=error_q_int_ant)
    #     sim.data.ctrl[:] = u
    #     # ctrl.q_nullspace = ctrl.inv_kinematics(sim)
    #     ctrl.step(sim, k)
    #     # sim.step()
    #     if simulate:
    #         viewer.render()
    #     k += 1
    #     if k >= ctrl.n_timesteps:  # and os.getenv('TESTING') is not None:
    #         break

    if plot_2d:
        ctrl.plots()

    # print(trajectory.extra_points)
    # print("biggest element = ", max_diag_element)
