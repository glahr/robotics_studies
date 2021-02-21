#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from controllers_utils import CtrlUtils, CtrlType
from draw import reference
from plots import displacement_plot
from soft_body_complete_analysis import soft_body
from soft_body_complete_analysis import reference_guide
from soft_body_complete_analysis import counting_words


# reference()

def load_model_mujoco(simulate, use_gravity): #, box=None
    # dict = {
    #     1: "assets/full_kuka_all_joints_soft_body_part_one",
    #     2: "assets/full_kuka_all_joints_soft_body_part_two",
    #     3: "assets/full_kuka_all_joints_soft_body_part_three",
    #     4: "assets/full_kuka_all_joints_soft_body_part_four",
    #     5: "assets/full_kuka_all_joints_soft_body_part_five",
    #     6: "assets/full_kuka_all_joints_soft_body_part_six",
    #     7: "assets/full_kuka_all_joints_soft_body_part_seven",
    #     8: "assets/full_kuka_all_joints_soft_body_part_eight",
    #     9: "assets/full_kuka_all_joints_soft_body_part_nine",
    #     10: "assets/full_kuka_all_joints"
    # }

    #model_name = dict.get(box)
    model_name = sf.xml_model()

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
    use_gravity = False  # If False, loads the model without gravity.

    plot_2d = False  # Show 2D plots at the end. But still need to fix after all new features.

    use_kd = True  # If True, use PD feedback. If False, just P.
    use_ki = True  # If True use a PID feedback. If False, just PD.
    simulation_time = 10  # [s]. This is the maximum time the robot will wait at the same position.

    # Choose with controller should be used. CtrlType.INV_DYNAMICS is the default
    # controller_type = CtrlType.INDEP_JOINTS
    # controller_type = CtrlType.INV_DYNAMICS
    # controller_type = CtrlType.INV_DYNAMICS_OP_SPACE

    reference_guide()
    box = int(input("Enter the number of box to be analyzed (number 10 are all boxes): "))
    sf = soft_body(box)
    try:
        open("displacement.txt", "r")
    except IOError:
        sf.creating_files()

    sim, viewer = load_model_mujoco(show_simulation, use_gravity) #, box=box
    sim.step()  # single step for operational space update
    ctrl = CtrlUtils(sim, simulation_time=simulation_time, use_gravity=use_gravity,
                     plot_2d=plot_2d, use_kd=use_kd, use_ki=use_ki)

    ctrl.get_pd_matrices()
    ctrl.Kp[5, 5] = 10
    ctrl.Kd[5, 5] = 10

    ctrl.Kp[4, 4] = 100
    ctrl.Kd[4, 4] = 20

    ctrl.Kp[3, 3] = 200
    ctrl.Kd[3, 3] = 20

    ctrl.Kp[1, 1] = 500
    ctrl.Kd[1, 1] = 30

    # print(ctrl.Kp[1, 1])
    # print(ctrl.Kd[1, 1])
    # print(sim.data.sensordata)

    # Inverse dynamics in joint space
    # qd = np.array([0, 0.461, 0, -0.817, 0, 0.69, 0])
    # POSITIONING
    # ctrl.use_ki = False
    # ctrl.controller_type = CtrlType.INDEP_JOINTS
    #qd = np.array([0, 0, 0, -np.pi / 2, -np.pi/2, 0, 0])
    # ctrl.move_to_joint_pos(sim, qd=qd, viewer=viewer)
    # qd[5] += np.pi/2
    # ctrl.move_to_joint_pos(sim, qd=qd, viewer=viewer)


    # INVERSE DYNAMICS
    ctrl.use_ki = False
    ctrl.get_pd_matrices()
    # ctrl.controller_type = CtrlType.IMPEDANCE
    qd = np.array([0, 0, 0, -np.pi / 2, -np.pi/2, np.pi / 2, 0])
    ctrl.move_to_joint_pos(sim, qd=qd, viewer=viewer)
    # xd = np.array([0.40001903, -0.24001999, 1.77999263])
    # xdmat = np.array([[9.98630075e-01, 5.23255792e-02, 8.01740344e-05],
    #                   [8.29898508e-05, -5.16406498e-05, -9.99999995e-01],
    #                   [-5.23255748e-02, 9.98630077e-01, -5.59123980e-05]])
    # ctrl.move_to_point(xd=xd, xdmat=xdmat, sim=sim, viewer=viewer)
    # qd = np.array([0, 0, -0.08, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
    # ctrl.move_to_joint_pos(sim, qd=qd, viewer=viewer)
    
    # print(ctrl.get_site_pose(sim))

    # IMPEDANCE CONTROL
    ctrl.controller_type = CtrlType.IMPEDANCE
    # ctrl.controller_type = CtrlType.INV_DYNAMICS_OP_SPACE
    xd, xdmat = ctrl.get_site_pose(sim)

    y_step = -0.0001
    k = 0


    # ctrl.K[0, 0] = 10
    # ctrl.B[0, 0] = 10
    #
    # ctrl.K[1, 1] = 10
    # ctrl.B[1, 1] = 10
    #
    # ctrl.K[2, 2] = 12
    # ctrl.B[2, 2] = 12

    ctrl.K[3, 3] = 3
    ctrl.B[3, 3] = 3

    ctrl.K[4, 4] = 5
    ctrl.B[4, 4] = 5

    # ctrl.K[5, 5] = 3
    # ctrl.B[5, 5] = 3

    '''
    Alguns pontos:
    
    - Caso vc queira ver o controle de impedância funcionando, vc pode só testar aqui as proximas três linhas.
    
    - Se der um erro NaN na aceleração, é porque o robô entrou no seu corpo flexível e isso gerou outras
    forças que vc não esperava. Reconsidere os deslocamentos.
    
    - Por fim, o controlador de impedância agora vc pode ir monitorando a força. Tem uma função que vc pode acessar como
    ctrl.get_ft_data(sim) que te retorna um np array como [fx, fy, fz, mx, my, mz]. Isso no sistema de coordenadas LOCAL
    do peg. Então vc tem que fazer logging da fz e do deslocamento em y pra ter os dados que vc quer.
    
    - Agora vc tem diversas matrizes que podem ajustar seu comportamento. O controlador de impedância implementado
    é um BK, depois podemos incluir M se for o caso. Pra alterar B e K vc acessa antes do ponto como ctrl.B e ctrl.K.
    Lembrando que ambos são matrizes 6x6. Vc pode brincar com isso se precisar, tornando mais rígido em certas direções.
    
    Have fun.
    
    Obs: recomendo vc rever a posição xyz dos corpos flexíveis, eles estão fora de centro e isso impacta em como o
    controlador de impedância se comporta.
    '''
    xd[1] += -0.03168
    ctrl.move_to_point(xd=xd, xdmat=xdmat, sim=sim, viewer=viewer)
    force = np.sqrt(np.power(ctrl.get_ft_data(sim)[0], 2) + np.power(ctrl.get_ft_data(sim)[1], 2)
                    + np.power(ctrl.get_ft_data(sim)[2], 2))
    # force = ctrl.get_ft_data(sim)[1]
    print(force)
    print(ctrl.get_ft_data(sim)[0])
    print(ctrl.get_ft_data(sim)[1])
    print(ctrl.get_ft_data(sim)[2])
    print("")
    # xd, xdmat = ctrl.get_site_pose(sim)

    # while k < 40:
    #     xd[1] += y_step
    #     ctrl.move_to_point(xd=xd, xdmat=xdmat, sim=sim, viewer=viewer)
    #     print(k)
    #     k += 1


    # xd, xdmat = ctrl.get_site_pose(sim)
    #
    # xd[0] += 0.05
    # xd[1] -= 0.05
    # print("aqui 1")
    # ctrl.move_to_joint_pos(sim, box=box, xd=xd, xdmat=xdmat)
    # print("aqui 2")
    # qd = np.array([0, 0, 0, -np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
    # ctrl.move_to_joint_pos(sim, qd=qd, viewer=viewer)
    # qd = np.array([0, -0.4, 0, -0.3, .5, 0.69, 0])
    # ctrl.move_to_joint_pos(qd, sim, viewer=viewer)
    # qd = np.array([0, 0.1, 0, -0.6, 0, 0.13, 0])
    # ctrl.move_to_joint_pos(qd, sim, viewer=viewer)
    # qd = np.array([np.pi/3, 0, 0, -np.pi/2, 0, 0, 0])
    # ctrl.move_to_joint_pos(qd, sim, viewer=viewer)
    # q0 = np.zeros(7)
    # ctrl.move_to_joint_pos(q0, sim, viewer=viewer)
    # print(sim.data.sensordata)
    # print(sim.data.sensordata[6:])
    # f = open("displacement.txt", "r")
    # y, t = counting_words(f)
    # print(f.tell())
    # p = f.tell()
    # f.seek((f.tell()-(y + t + 1)), 0)
    # print(f.read(y))
    # # print(f.read(1))
    # print(f.read(t))
    # f.seek(p, 0)
    # print(f.tell())
    # f.seek(116,0)
    # y, t = counting_words(f)
    # print(f.tell())
    # p = f.tell()
    # f.seek((f.tell() - (y + t + 1)), 0)
    # print(f.read(y))
    # # print(f.read(1))
    # print(f.read(t))
    # f.seek(p, 0)
    # print(f.tell())
    # y, t = counting_words(f)
    # print(f.tell())
    # y, t = counting_words(f)
    # print(f.tell())
    # y, t = counting_words(f)
    # print(f.tell())
    # f.close()
    # if KeyboardInterrupt:
    #     sf.displacement_plot()

    sf.displacement_plot()


    # Inverse dynamics in joint space with operational space
    # xd, xdmat = ctrl.get_site_pose(sim)
    #
    # xd[0] -= 0.05
    # ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)
    # xd[1] += 0.05
    # ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)
    # xd[0] += 0.05
    # ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)
    # xd[1] -= 0.05
    # ctrl.move_to_joint_pos(sim, xd=xd, xdmat=xdmat, viewer=viewer)

    # Operational space control
    # ctrl.controller_type = CtrlType.INV_DYNAMICS_OP_SPACE
    #
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
