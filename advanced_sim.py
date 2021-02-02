#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from numpy import pi
from controllers_utils import CtrlUtils, CtrlType


def load_model_mujoco(simulate, use_gravity):
    model_name = "assets/full_kuka_all_joints"
    if use_gravity:
        model_name += "_gravity"
    model_xml = load_model_from_path(model_name + '.xml')
    sim = MjSim(model_xml)
    if simulate:
        viewer = MjViewer(sim)
    else:
        viewer = None
    return sim, viewer


def my_rand_pos():
    # qd = np.random.rand(7) - 0.5
    # qd = [(np.random.rand()-1.5)*1, np.random.rand(), (np.random.rand()-0.5)*4,
    #       -np.pi/2 + (np.random.rand()-0.5)*3, (np.random.rand()-0.5)*3, np.pi / 2++ (np.random.rand()-0.5)*3, 0]
    # qd = np.array([qi+np.random.rand() for qi in q])

    if np.random.rand() < 0.5:
        qd = np.array([pi/4, pi/2, pi/2, pi/2, pi/2, -pi/2, 0])
    else:
        qd = np.array([-pi/4, pi/2, pi/2, -pi/2, -pi/2, pi/2, 0])

    qd = qd + np.array([np.random.rand()-0.5 for _ in range(7)])

    return qd


if __name__ == '__main__':

    show_simulation = True # If True, plot the 3D simulation at runtime.
    use_gravity = True  # If False, loads the model without gravity.
    plot_2d = True  # Show 2D plots at the end. But still need to fix after all new features.
    use_kd = True  # If True, use PD feedback. If False, just P.
    use_ki = True  # If True use a PID feedback. If False, just PD.
    simulation_time = 10  # [s]. This is the maximum time the robot will wait at the same position.

    # Choose with controller should be used. CtrlType.INV_DYNAMICS is the default
    # controller_type = CtrlType.INDEP_JOINTS
    # controller_type = CtrlType.INV_DYNAMICS
    # controller_type = CtrlType.INV_DYNAMICS_OP_SPACE

    sim, viewer = load_model_mujoco(show_simulation, use_gravity)
    sim.step()  # single step for operational space update
    ctrl = CtrlUtils(sim, simulation_time=simulation_time, use_gravity=use_gravity,
                     plot_2d=plot_2d, use_kd=use_kd, use_ki=use_ki)

    # Inverse dynamics in joint space
    # qd = np.array([0, 0.461, 0, -0.817, 0, 0.69, 0])
    qd = np.array([0, 0, 0, -np.pi / 2, 0, np.pi/2, 0])
    ctrl.move_to_joint_pos(sim, qd=qd, viewer=viewer)


    # standard trajectory
    xd, xdmat = ctrl.get_site_pose(sim)
    # sim.data.mocap_pos[0] = np.array([0.4, 0.5, 1.35])
    ctrl.move_to_point(sim=sim, xd=xd, xdmat=xdmat, viewer=viewer)
    xd[0] += 0.3
    xd[1] += 0.3
    xd[2] -= 0.45
    # ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)
    ctrl.move_to_point(sim=sim, xd=xd, xdmat=xdmat, viewer=viewer)
    xd[1] -= 0.44
    # ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)







    # xd[1] += 0.05
    # ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)
    # xd[0] += 0.05
    # ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)
    # xd[1] -= 0.05
    # ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)

    # Operational space control
    # ctrl.controller_type = CtrlType.INV_DYNAMICS_OP_SPACE

    # xd, xdmat = ctrl.get_site_pose(sim)
    # xd[0] -= 0.05
    # ctrl.move_to_point(xd, xdmat, sim, viewer=viewer)
    # xd[1] += 0.05
    # ctrl.move_to_point(xd, xdmat, sim, viewer=viewer)
    # xd[0] += 0.05
    # ctrl.move_to_point(xd, xdmat, sim, viewer=viewer)
    # xd[1] -= 0.05
    # ctrl.move_to_point(xd, xdmat, sim, viewer=viewer)

    if plot_2d:
        ctrl.plots()
