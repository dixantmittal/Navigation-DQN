import gym
import numpy

from simulator.ISimulator import ISimulator


class CarlaSimulator(ISimulator):
    def __init__(self):
        self.env = gym.make('AirRaid-v0')

    def reset(self):
        return self.env.reset()

    def step(self, a):
        return self.env.step(a)

    def nActions(self):
        return self.env.action_space.n

    def dState(self):
        return self.env.observation_space.shape

    def sampleAction(self):
        return numpy.random.randint(self.nActions())

    def prettifyState(self, rawState):
        H, W, C = self.dState()
        return rawState.reshape(-1, H, W, C)

    def __del__(self):
        self.env.close()
