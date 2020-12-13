#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
from gym_kuka_mujoco.utils.kinematics import forwardKin, forwardKinJacobian, forwardKinSite, forwardKinJacobianSite
import mujoco_py
import numpy as np
from controllers_utils import CtrlUtils


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

    simulation_time = 6  # s

    # qd = np.array([0, 0.461, 0, -0.817, 0, 0.69, 0])
    # qd = np.array([0, 0, 0, 0, 0, 0, 0])
    qd = np.array([0, 0, 0, -np.pi/2, -np.pi/2, 0, 0])

    sim, viewer = load_model_mujoco(simulate, use_gravity)
    sim.forward()
    ctrl = CtrlUtils(sim, simulation_time, use_gravity, plot_2d, use_kd, use_ki)
    ctrl.controller_type = 'independent_joints'

    if ctrl.controller_type == 'independent_joints':
        ctrl.qd = qd
        ctrl.kp = 11
        ctrl.lambda_H = 3.5
    k = 1

    tf = 1  # spline final time
    ctrl.trajectory_gen_joints(qd, tf, traj='spline5')

    eps = 1*np.pi/180

    while True:

        # if (np.absolute(erro_q) < eps).all():
        #     qd = np.array([0, 0, 3/2*np.pi/2, 0, 0, -np.pi/2, 0])
        # print("tolerancia " + str(sim.data.time))

        ctrl.calculate_errors(sim, k)

        u = ctrl.ctrl_action(sim)  # , erro_q, erro_v, error_q_int_ant=error_q_int_ant)

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

    # print("biggest element = ", max_diag_element)
