import argparse

import torch

import Logger
from QNetwork import QNetwork
from SimulatorFactory import SimulatorFactory

parser = argparse.ArgumentParser()
parser.add_argument('--simulator', dest='simulator', help='Simulator class name', required=True)
parser.add_argument('--network', dest='networkPath', help='Path to the saved network file', required=True)
parser.add_argument('--device', dest='device', help='[cpu, cuda]', default='cpu')
parser.add_argument('--logger', dest='logger', help='Logging sensitivity', default='info')
args = parser.parse_args()

Logger.setLevel(args.logger)
logger = Logger.logger

logger.info('List of Parameters:\n'
            'simulator: %s\n'
            'networkPath: %s\n'
            'device: %s\n'
            'logger: %s\n',
            args.simulator,
            args.networkPath,
            args.device,
            args.logger)

if __name__ == '__main__':
    simulator = SimulatorFactory.getInstance(args.simulator)
    network = QNetwork(simulator.dState(), simulator.nActions())
    network.load(args.networkPath)

    network.to(args.device)

    logger.info('Starting simulation')
    state = simulator.reset()
    done = False
    episodeLength = 0
    while not done:
        episodeLength += 1

        action = network(torch.Tensor(state).to(args.device)).argmax().item()
        state, reward, done, _ = simulator.step(action)

        logger.info('Reward received: %s', reward)

    logger.info('Episode complete. Length: %s', episodeLength)
