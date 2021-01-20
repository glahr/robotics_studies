# robotics_studies
Repo for initial studies in robotics kinematics and dynamics control with Mujoco.

## Running
Four codes are available: 
1) ```classic_control_two_link.py```: simulates a planar RR robot. All code is at this file. Good start.
2) ```main_sim.py```: a self-contained inverse dynamics controller with Kuka iiwa robot simulation. Intermediate (used for the Mujoco Workshop 2020).
3) ```advanced_sim.py```: full code with three different controller and trajectory planning. All codes are in ```controllers_utils.py``` for development and use. Advanced users should get from here. Create your own sim file and use this examples for coding. Contributions are welcome!
4) ```camera_rendering_example.py```: minimal example of a robotic arm with embedded camera using mujoco-py.

## Steps for starting your robotics studies

1) Activate and deactivate Mujoco viewer
2) Access Mujoco data info during simulation
3) Calculate forward kinematics and compare with Mujoco's ```xpos```
4) Calculate the Geometric Jacobian and compare with Mujoco's ```.jacp && .jacr```
5) Obtain cartesian velocity through xvel and calculated Jacobian
6) Create a joint position PD controller (simulate at Matlab to find best params)
7) Apply inverse kinematics and move in Cartesian coordinates
8) 


## Troubleshooting
### Can't run Mujoco in debug mode with Pycharm
Even though it suggest to add ```LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/.mujoco/mujoco200/bin``` to .bashrc, Pycharm in debug mode seems to not load it correclty.

The solution is to add it to into ``` Edit configurations->Environment Variables``` click on ```+``` and then on the left panel add ```LD_LIBRARY_PATH```, on the left ```$LD_LIBRARY_PATH:/home/$USER/.mujoco/mujoco200/bin```.

## Contributors and FAQ
Gustavo Lahr - glahr - gustavo.lahr@usp.br

Henrique Garcia - griloHBG
