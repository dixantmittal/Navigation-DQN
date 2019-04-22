import argparse

import torch

import carla
from q_network import QNetwork
from simulator import Simulator

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
args = parser.parse_args()

# Initialise simulator and network
simulator = Simulator(host = args.sim_ip, n_vehicles = int(args.n_vehicles))
network = QNetwork()

# Load the saved network file
network.load(args.network_path)

# Check if cuda is avialable
to_cuda = torch.cuda.is_available()

if to_cuda:
    network = network.cuda()

try:
    terminate = False
    state = simulator.reset()

    while not terminate:
        # Convert state to Tensor
        state = torch.cuda.FloatTensor(state) if to_cuda else torch.FloatTensor(state)

        # Find the Q-value
        Q = network(state.unsqueeze(0))

        # Select the best action
        action = Q.argmax().item()

        # Perform the action
        result = simulator.step(action)

        # Observe results
        state = result['state']
        terminate = result['terminate']

except KeyboardInterrupt:
    # Destroy simulator before quitting
    simulator.destroy()
