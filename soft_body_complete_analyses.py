import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from controllers_utils import CtrlUtils, CtrlType
import pygame


def reference_guide():
    pygame.init()

    # logo = pygame.image.load("logo32x32.png")
    # pygame.display.set_icon(logo)
    image = pygame.image.load("Drawing.png")
    pygame.display.set_caption("Reference image")

    screen = pygame.display.set_mode((450, 450))
    screen.blit(image, (0, 0))

    pygame.display.flip()

    running = True

    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False


class soft_body:

    def __init__(self, box=None):
        self.box = box

    def xml_model(self):
        dict = {
            1: "assets/full_kuka_all_joints_soft_body_part_one",
            2: "assets/full_kuka_all_joints_soft_body_part_two",
            3: "assets/full_kuka_all_joints_soft_body_part_three",
            4: "assets/full_kuka_all_joints_soft_body_part_four",
            5: "assets/full_kuka_all_joints_soft_body_part_five",
            6: "assets/full_kuka_all_joints_soft_body_part_six",
            7: "assets/full_kuka_all_joints_soft_body_part_seven",
            8: "assets/full_kuka_all_joints_soft_body_part_eight",
            9: "assets/full_kuka_all_joints_soft_body_part_nine",
            10: "assets/full_kuka_all_joints"
        }

        model_name = dict.get(self.box)
        return model_name

    def displacement_plot(self, displacement, time,):
        title = "stiffness of" + " " + str(self.box)
        plt.plot(time, displacement, 'b', linewidth=2.0)
        # plt.setp(color='b', linewidth=2.0)
        plt.xlabel('time of iteration')
        plt.ylabel('displacement')
        plt.title(title)

        plt.grid(True)
        plt.savefig(title)
        plt.show()