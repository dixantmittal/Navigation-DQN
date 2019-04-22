import argparse
import time

import carla
from q_network import QNetwork
from simulator import Simulator
from dqn import DQNTrainer

# Clean the environment
client = carla.Client('localhost', 2000)
world = client.get_world()
for actor in world.get_actors():
    if 'vehicle' in actor.type_id or 'sensor' in actor.type_id:
        actor.destroy()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ip',
                    dest = 'sim_ip',
                    default = 'localhost',
                    help = 'Simulator IP')
parser.add_argument('--network',
                    dest = 'network_path',
                    default = None,
                    help = 'Network file path')
parser.add_argument('--n_vehicles',
                    dest = 'n_vehicles',
                    default = 5)
parser.add_argument('--lr',
                    dest = 'lr',
                    default = 1e-4)
parser.add_argument('--eps',
                    dest = 'eps',
                    default = 0.999)
args = parser.parse_args()

# Initialise the simulator
simulator = Simulator(host = args.sim_ip, n_vehicles = int(args.n_vehicles))

# Initialise the network
network = QNetwork()

# Load network if a pre-trained exists
if args.network_path is not None:
    network.load(args.network_path)

# Load DQN
trainer = DQNTrainer(simulator, network)

# Training phase
try:
    trainer.train(_batch_size = 128,
                  epoch = 50000,
                  _learning_rate = float(args.lr),
                  _target_update_frequency = 40,
                  _plot_frequency = 10,
                  eps = float(args.eps),
                  gamma = 0.9,
                  verbose = True)
except KeyboardInterrupt:
    # Save the trained network
    network.save(str(time.time()) + '.pth')

    # Destroy the simulator before quitting
    simulator.destroy()
