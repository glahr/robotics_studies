import roboticstoolbox as rtb
import numpy as np

iiwa = rtb.models.DH.LWR4()

qd = np.array([0, 0.461, 0, -0.817, 0, 0.69, 0])

iiwa.tool.R[:] = np.eye(3)  # TODO: change tool length

T_new = iiwa.base

T_new.t[2] = 0.36
iiwa.base = T_new

# links = [rtb.RevoluteDH(d=0.36)]
#
# for link in iiwa.links:
#     links.append(link)
#
# new_robot = rtb.DHRobot(links)

iiwa.plot(np.zeros(7))
iiwa.plot(qd)

# new_robot.plot(qd)