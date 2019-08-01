import threading
import time

import numpy
import torch

import Logger
from ExperienceCollector import ExperienceCollector
from ReplayMemory import ReplayMemory
from SimulatorFactory import SimulatorFactory

logger = Logger.logger


def to_one_hot(indices, n_classes):
    one_hot = torch.zeros(len(indices), n_classes)
    one_hot[torch.arange(len(indices)), indices.to(torch.long).squeeze()] = 1
    return one_hot


class DQN(object):

    def __init__(self, network):
        self.experienceCollectors = []

        self.threadSync = None
        self.stopSyncThread = False

        self.network = network

    def syncNetwork(self):
        while not self.stopSyncThread:
            for experienceCollector in self.experienceCollectors:
                logger.info('Syncing policy network with %s', experienceCollector.name)
                experienceCollector.lock.acquire()
                try:
                    experienceCollector.network = self.network.copy()
                finally:
                    experienceCollector.lock.release()
            time.sleep(2)

        logger.info('Stopped thread: Network Sync')

    def train(self, args):
        # Parameters
        gamma = args.gamma
        device = args.device

        simulator = SimulatorFactory.getInstance(args.simulator)
        dStates = simulator.dState()
        nStates = numpy.prod(dStates)
        nActions = simulator.nActions()

        # Initialise Metrics
        metrics = {
            'test_set': [],
            'best_test_performance': -numpy.inf
        }

        self.experienceCollectors = [ExperienceCollector(i, self.network, args) for i in range(args.threads)]
        self.threadSync = threading.Thread(target=self.syncNetwork, name='SyncThread')
        self.threadSync.start()
        logger.info('Started thread: Network Sync')

        # Wait while ReplayMemory collects some experiences.
        while ReplayMemory.memoryEmpty:
            time.sleep(0.1)

        # initialise a test set
        logger.info('Loading test set')
        test = []
        for i in range(args.testSize):
            test.append(simulator.reset())
        logger.info('Test set loaded!')
        test = torch.Tensor(test).to(device)

        policyNetwork = self.network.to(device)
        targetNetwork = policyNetwork.copy().to(device)

        # initialise optimiser and loss function
        optimiser = torch.optim.Adam(policyNetwork.parameters(), lr=args.lr, weight_decay=1e-4)
        lossFn = torch.nn.MSELoss()

        itr = 0
        while args.itr == 0 or itr < args.itr:
            itr += 1

            if itr % args.frequency == 0:
                targetNetwork = policyNetwork.copy().to(device)

            # OPTIMIZE POLICY
            batch = ReplayMemory.sample(args.batchSize)

            # slice them to get state and actions
            batch = torch.Tensor(batch).to(device)
            state, action, next_state, reward, terminate = torch.split(batch, [nStates, 1, nStates, 1, 1], dim=1)

            action = to_one_hot(action, nActions).to(device)

            state = simulator.prettifyState(state).to(device)
            next_state = simulator.prettifyState(next_state).to(device)

            # find the target value
            target = reward + terminate * gamma * targetNetwork(next_state).max(dim=1)[0].unsqueeze(dim=1)

            # Calculate Q value
            predicted = (policyNetwork(state) * action).sum(dim=1).unsqueeze(dim=1)

            # find loss
            loss = lossFn(predicted, target)

            # Backprop
            optimiser.zero_grad()
            loss.backward()

            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad.clip_grad_value_(policyNetwork.parameters(), 5)
            optimiser.step()

            # Terminate episode if contribution of next state is small.
            # Store Evaluation Metrics
            metrics['test_set'].append(policyNetwork(test).max(dim=1)[0].mean().item())

            # Print statistics
            logger.info('[Iteration: %s] Test Q-values: %s', itr, metrics['test_set'][-1])

            # Checkpoints
            if metrics['test_set'][-1] > metrics['best_test_performance']:
                metrics['best_test_performance'] = metrics['test_set'][-1]

                policyNetwork.save(args.networkPath)
                if args.checkpoints:
                    policyNetwork.save('checkpoints/Q_network_{}.pth'.format(metrics['test_set'][-1]))

    def stop(self):
        for collector in self.experienceCollectors:
            collector.stop()

        self.stopSyncThread = True
        if self.threadSync is not None:
            self.threadSync.join()
