# DRL_Environment_Carla
We constructed a DRL interactive environment based on the [Carla simulatior](https://github.com/carla-simulator/carla).  
The original scene was to start from the highway ramp and drive along the highway.
If you need to complete other scene tasks, please reset the starting and ending points according to the scene conditions.  

In this code, the reinforcement learning algorithm used is SAC and experience buffering is used for training.
Since our paper is still under review, the main code is provided for reference, while the environment and data processing code will be updated after the paper is accepted.

# Requirements
- Python => 3.7
- Torch => 11.6
- Gym => 0.12.5
- Pygame => 1.9.6
- Tensorboard => 2.11.2
- Carla => 0.9.10

# Usage
Carla should be activated first before starting training.  
```
$ python start_carla.py
```
TO run the SAC algorithm based on the environment for training:
```
$ python run_training.py
```

# Results
To run Tensorboard on the terminal, view the training situation and related metrcis in real-time.  
**Please note:**  
The code will create a new folder and store the relevant data in order when running the * *run_training.py* *.   
So xxx in exp_xxx represents the number of runs * *run_training.py* *.  
```
$ tensorboard --logdir=./exp-SAC/exp_xxx
```
# Citation
If you find this code useful in your research, please cite our paper, thank you.  
