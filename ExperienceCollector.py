import threading

import numpy
import torch

import Logger
from ReplayMemory import ReplayMemory
from SimulatorFactory import SimulatorFactory

logger = Logger.logger


class ExperienceCollector(object):
    def __init__(self, id, network, args):
        self.network = network
        self.args = args
        self.name = 'ExperienceCollector_{}'.format(id)

        self.stopThread = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.collect, name=self.name)
        self.thread.start()
        logger.info('Started thread: %s', self.name)

    def collect(self):
        # Parameters
        eps = self.args.eps
        gamma = self.args.gamma
        device = self.args.device

        simulator = SimulatorFactory.getInstance(self.args.simulator)
        buffer = ReplayMemory(self.args.memory)

        itr = 0
        while not self.stopThread:
            eps = max(0.1, eps ** itr)

            done = False
            episodeReward = 0
            episodeLength = 0

            self.lock.acquire()
            try:
                policyNetwork = self.network.to(device)

                # Reset simulator for new episode
                logger.debug('Starting new episode')

                state = simulator.reset()
                while not done and not self.stopThread:
                    action = simulator.sampleAction()
                    if numpy.random.rand() > eps:
                        action = policyNetwork(torch.Tensor(state).to(device)).argmax().item()

                    # take action and get next state
                    nextState, reward, done, _ = simulator.step(action)

                    # store into experience memory
                    buffer.push(state, action, nextState, reward, int(not done))

                    state = nextState
                    episodeReward += reward * gamma ** episodeLength
                    episodeLength += 1

                    if gamma ** episodeLength < 0.1:
                        break

                logger.debug('Episode Length: %s \tEpisode Reward: %s', episodeLength, episodeReward)
            finally:
                self.lock.release()

        buffer.stop()
        logger.info('Stopped thread: %s', self.name)

    def stop(self):
        self.stopThread = True
        self.thread.join()
