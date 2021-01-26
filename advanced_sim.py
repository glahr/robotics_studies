#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
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
    qd = np.array([0, 0.461, 0, -0.817, 0, 0.69, 0])
    # qd = np.array([0, 0, 0, -np.pi / 2, 0, np.pi/2, 0])
    ctrl.move_to_joint_pos(sim, qd=qd, viewer=viewer)
    # qd = np.array([0, -0.4, 0, -0.3, .5, 0.69, 0])
    # ctrl.move_to_joint_pos(sim, qd=qd, viewer=viewer)
    # qd = np.array([0, 0.1, 0, -0.6, 0, 0.13, 0])
    # ctrl.move_to_joint_pos(sim, qd=qd, viewer=viewer)
    # qd = np.array([np.pi/3, 0, 0, -np.pi/2, 0, 0, 0])
    # ctrl.move_to_joint_pos(sim, qd=qd, viewer=viewer)
    # q0 = np.zeros(7)
    # ctrl.move_to_joint_pos(sim, qd=qd, viewer=viewer)

    # Inverse dynamics in joint space with operational space
    # xd, xdmat = ctrl.get_site_pose(sim)
    # xd = sim.data.get_site_xpos(ctrl.name_hole_top)
    # xdmat = sim.data.get_site_xmat(ctrl.name_hole_top)

    # xd[0] -= 0.05
    # ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)
    # xd[1] += 0.05
    # ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)
    # xd[0] += 0.05
    # ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)
    # xd[1] -= 0.05
    # ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)
    # xd[2] -= 0.2
    # ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)

    # Operational space control
    ctrl.controller_type = CtrlType.INV_DYNAMICS_OP_SPACE
    xd = np.array([4.01863413e-01, -4.95389989e-05, 1.53277615e+00])
    xdmat = np.array([[ 0.99995499, -0.00210169,  0.00925189],
                      [-0.00209056, -0.99999708, -0.0012129 ],
                      [ 0.00925441,  0.0011935 , -0.99995646]])
    #
    # xd, xdmat = ctrl.get_site_pose(sim)
    xd[0] -= 0.05
    ctrl.move_to_point(xd, xdmat, sim, viewer=viewer)
    xd[1] += 0.05
    theta = np.pi/8
    ctrl.move_to_point(xd, xdmat, sim, viewer=viewer)
    xd[0] += 0.05
    ctrl.move_to_point(xd, xdmat.dot([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]]), sim, viewer=viewer)
    xd[1] -= 0.05
    ctrl.move_to_point(xd, xdmat, sim, viewer=viewer)

    if plot_2d:
        ctrl.plots()
