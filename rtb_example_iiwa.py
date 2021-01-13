import roboticstoolbox as rtb
import numpy as np
import spatialmath as smath

iiwa = rtb.models.DH.LWR4()

qd = np.array([0, 0.461, 0, -0.817, 0, 0.69, 0])
qd2 = np.array([np.pi/3, 0, 0, -np.pi/2, 0, 0, 0])
q0 = np.zeros(7)

# adjusting links according to datasheet for R820
iiwa.links[2].d = 0.42
iiwa.links[4].d = 0.4

# fixing tool
iiwa.tool.R[:] = np.eye(3)  # TODO: change tool length
iiwa.tool.t[2] = 0.159 + 0.126 - 0.04498

# fixing base
T_new = iiwa.base
T_new.t[2] = 0.36
T_new.R[:] = smath.SO3.Rz(np.pi) * T_new.R
iiwa.base = T_new

# links = [rtb.RevoluteDH(d=0.36)]
#
# for link in iiwa.links:
#     links.append(link)
#
# new_robot = rtb.DHRobot(links)

iiwa.plot(np.zeros(7))
iiwa.plot(qd)
iiwa.plot(qd2)

# new_robot.plot(qd)