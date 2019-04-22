# Navigation-DQN
A Deep Q-Network trained to safely guide the autonomous car to cross a 4-way intersection in a complex traffic environment.

# Requirements
* Python 3
* PyTorch (https://pytorch.org)
* Carla (http://carla.org)

# Installation
```shell
$ python3 -m venv .env
$ source .env/bin/activate
$ pip install -r requirements.txt
```

# Simulator
* Download Carla simulator binary files (https://github.com/carla-simulator/carla/releases). 
* Extract the .tar file to `<location>`
* Start the simulator by
```shell
$ ./<location>/CarlaUE4.sh Town03
```
* To go to simulation site, go towards the left side from the spawned location (using keys WASDQE and mouse pointer)

# Training
If you have a pretrained file
```shell
$ source .env/bin/activate
$ python3 trainer.py. --n_vehicles 10 --network <path to file>
```
else
```shell
$ source .env/bin/activate
$ python3 trainer.py. --n_vehicles 10
```

# Testing/Demo
```shell
$ source .env/bin/activate
$ python3 tester.py. --n_vehicles 10 --network <path to file>
```

# GPU
It is recommended to use GPU for training and inference.
