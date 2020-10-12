# robotics_studies
Repo for initial studies in robotics kinematics and dynamics control with Mujoco.

## Steps

1) Activate and deactivate Mujoco viewer
2) Access Mujoco data info during simulation
3) Calculate forward kinematics and compare with Mujoco's ```xpos```
4) Calculate the Geometric Jacobian and compare with Mujoco's ```.jacp && .jacr```
5) Obtain cartesian velocity through xvel and calculated Jacobian
6) Create a joint position PD controller (simulate at Matlab to find best params)
7) 


## Troubleshooting
### Can't run Mujoco in debug mode with Pycharm
Even though it suggest to add ```LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/glahr/.mujoco/mujoco200/bin``` to .bashrc, Pycharm in debug mode seems to not load it correclty.

The solution is to add it to into ``` Edit configurations->Environment Variables``` click on ```+``` and then on the left panel add ```LD_LIBRARY_PATH```, on the left ```$LD_LIBRARY_PATH:/home/glahr/.mujoco/mujoco200/bin```.
