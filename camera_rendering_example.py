from mujoco_py import MjSim, MjViewer, MjRenderContextOffscreen
import mujoco_py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter 
from functools import partial
import cv2 
import time

# TODO: improve this guy rendering. It is not working.
# def render_matplotlib():
#     fig = plt.figure(1)
#     ax = fig.add_subplot(111)
#     ax.set_title("My Title")
#     im = ax.imshow(np.zeros((width, height, 3)))  # Blank starting image
#     fig.show()
#     im.axes.figure.canvas.draw()
#
#     # vieweroff.render(width, height, camera_id=0)
#     # rgb = vieweroff.read_pixels(width, height)[0]
#     rgb = np.random.random((256, 256, 3))
#     im.set_data(rgb)
#     ax.set_title(str(t))
#     im.axes.figure.canvas.draw()

def render_opencv():
    # camera #1
    vieweroff.render(width, height, camera_id=0)
    rgb = vieweroff.read_pixels(width, height)[0]
    bgr = np.flipud(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imshow('teste1', bgr)

    # camera #2
    vieweroff.render(width, height, camera_id=1)  # if camera_id=None, camera is not rendered
    rgb = vieweroff.read_pixels(width, height)[0]
    bgr = np.flipud(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imshow('teste2', bgr)

    cv2.waitKey(1)


model = mujoco_py.load_model_from_path("./assets/my-model-rrr.xml")

''' ATTENTION: if you choose to use Mujoco's default viewer, you can't see the rendering of the cameras!'''

sim = MjSim(model)
vieweroff = MjRenderContextOffscreen(sim,0)

# controller and simulation params
t = 0
qpos_ref = np.array([-2, -1, 2])
qvel_ref = np.array([0, 0, 0])
kp = 1000
kv = 500

sim.model.opt.gravity[:] = np.array([0, 0, 0])  # just to make simulation easier :)

width, height = 800, 480

t_ini = time.time()

try:

    while True:

        # robot controller
        qpos_cur = sim.data.qpos
        qvel_cur = sim.data.qvel
        qpos_error = qpos_ref - qpos_cur
        qvel_error = qvel_ref - qvel_cur
        ctrl = qpos_error*kp + qvel_error*kv
        sim.step()

        # sim.render(width, height, camera_name="camera-ee", depth=False, mode='offscreen', device_id=0)
        # viewer.render()

        render_opencv()
        
        t = t + 1
        sim.data.ctrl[:] = ctrl

        if t > 1000:
            break

except KeyboardInterrupt:
    print("saindo")

t_total = time.time() - t_ini

FPS = t / t_total

print(FPS, "fps")