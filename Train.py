import argparse
import os

import Logger
from SimulatorFactory import SimulatorFactory
from DQN import DQN
from QNetwork import QNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--simulator', dest='simulator', help='Simulator class name', required=True)
parser.add_argument('--network', dest='networkPath', help='Path to the save trained network', required=True)
parser.add_argument('--vehicles', dest='nVehicles', help='Number of Vehicles', default=10, type=int)
parser.add_argument('--host', dest='host', help='Host address of the carla server', default='localhost')
parser.add_argument('--port', dest='port', help='Port of the carla server', default=2000, type=int)
parser.add_argument('--lr', dest='lr', default=1e-4, help='', type=float)
parser.add_argument('--eps', dest='eps', default=0.999, type=float)
parser.add_argument('--batch', dest='batchSize', default=32, type=int)
parser.add_argument('--itr', dest='itr', default=0, type=int, help='Number of iterations for training [0 for infinite]')
parser.add_argument('--threads', dest='threads', default=4, type=int)
parser.add_argument('--gamma', dest='gamma', default=0.99, type=float)
parser.add_argument('--frequency', dest='frequency', default=50, type=int)
parser.add_argument('--memory', dest='memory', default=10000, type=int, help='Buffer size (in number of experiences)')
parser.add_argument('--logger', dest='logger', help='Logging sensitivity', default='info')
parser.add_argument('--test_size', dest='testSize', help='Size of test set', default=1, type=int)
parser.add_argument('--device', dest='device', help='[cpu, cuda]', default='cpu')
parser.add_argument('--checkpoints', dest='checkpoints', action='store_true', default=False, help='store checkpoints')
args = parser.parse_args()

Logger.setLevel(args.logger)

logger = Logger.logger

logger.info('List of Parameters:\n'
            'simulator: %s\n'
            'networkPath: %s\n'
            'lr: %s\n'
            'batchSize: %s\n'
            'itr: %s\n'
            'eps: %s\n'
            'gamma: %s\n'
            'memory: %s\n'
            'frequency: %s\n'
            'testsize: %s\n'
            'device: %s\n'
            'threads:%s\n'
            'checkpoints:%s\n'
            'logger: %s\n',
            args.simulator,
            args.networkPath,
            args.lr,
            args.batchSize,
            args.itr,
            args.eps,
            args.gamma,
            args.memory,
            args.frequency,
            args.testSize,
            args.device,
            args.threads,
            args.checkpoints,
            args.logger)

if __name__ == '__main__':
    if args.checkpoints and not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    simulator = SimulatorFactory.getInstance(args.simulator, args)
    trainer = DQN(QNetwork(simulator.dState(), simulator.nActions(), args))
    try:
        logger.info('Starting training.')
        trainer.train(args)
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt received. Trying to stop threads.')
    finally:
        trainer.stop()
        simulator.destroy()
