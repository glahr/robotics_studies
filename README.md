# robotics_studies
Repo for initial studies in robotics kinematics and dynamics control with Mujoco.

## Steps

1) Activate and deactivate Mujoco viewer
2) Access Mujoco data info during simulation
3) Calculate forward kinematics and compare with Mujoco
4) 


## Troubleshooting
### Can't run Mujoco in debug mode with Pycharm
Even though it suggest to add ```LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/glahr/.mujoco/mujoco200/bin``` to .bashrc, Pycharm in debug mode seems to not load it correclty.

The solution is to add it to into ``` Edit configurations->Environment Variables``` click on ```+``` and then on the left panel add ```LD_LIBRARY_PATH```, on the left ```$LD_LIBRARY_PATH:/home/glahr/.mujoco/mujoco200/bin```.
