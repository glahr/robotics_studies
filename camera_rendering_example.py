from mujoco_py import MjSim, MjViewer, MjRenderContextOffscreen
import mujoco_py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter 
from functools import partial
import cv2 
import time

model = mujoco_py.load_model_from_path("./my-model-rrr.xml")

sim = MjSim(model)
#viewer = MjViewer(sim)
vieweroff = MjRenderContextOffscreen(sim,0)

t = 0

qpos_ref = np.array([-2, -1, 2])
qvel_ref = np.array([0, 0, 0])

kp = 1000
kv = 500

sim.model.opt.gravity[:] = np.array([0, 0, 0])

# fig = plt.figure( 1 )
# ax = fig.add_subplot( 111 )
# ax.set_title("My Title")

width, height = 800, 480

# im = ax.imshow( np.zeros( ( width, height, 3 ) ) ) # Blank starting image
# fig.show()
# im.axes.figure.canvas.draw()

t_ini = time.time()

try:

    while True:

        qpos_cur = sim.data.qpos
        qvel_cur = sim.data.qvel

        qpos_error = qpos_ref - qpos_cur
        qvel_error = qvel_ref - qvel_cur

        ctrl = qpos_error*kp + qvel_error*kv
        
        sim.step()
        #sim.render(width, height, camera_name="camera-ee", depth=False, mode='offscreen', device_id=0)
        #viewer.render() 
        vieweroff.render(width, height, camera_id=0)
        #rgb = np.flipud(vieweroff.read_pixels(width, height)[0]/255)
        #rgb = np.full((300,300,3), 125, dtype=np.uint8)
        #rgb= np.random.random( ( 256, 256, 3 ) )
        rgb = vieweroff.read_pixels(width, height)[0]
        #im.set_data( rgb )
        #ax.set_title( str( t ) )
        #im.axes.figure.canvas.draw()
        bgr = np.flipud(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow('teste1', bgr)

        vieweroff.render(width, height, camera_id=1)
        rgb = vieweroff.read_pixels(width, height)[0]
        bgr = np.flipud(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow('teste2', bgr)
        
        vieweroff.render(width, height, camera_id=None)
        rgb = vieweroff.read_pixels(width, height)[0]
        bgr = np.flipud(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow('teste3', bgr)
        
        cv2.waitKey(1)
        
        t = t + 1
        sim.data.ctrl[:] = ctrl
        # sim.data.ctrl[:] = np.array([np.sin(.1*t), np.cos(.1*t), np.sin(.1*t)*np.cos(.01*t)])*20000

        if t > 1000:
            break

except KeyboardInterrupt:
    print("saindo")

t_total = time.time() - t_ini

FPS = t_total / t

print(FPS, "fps")