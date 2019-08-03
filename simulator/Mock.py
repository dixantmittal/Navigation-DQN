import numpy


class Mock(object):

    def __init__(self, args):
        self.args = args
        self.rand = None

    def reset(self):
        self.rand = numpy.random.randint(10)
        return numpy.random.randint(low=10, high=40, size=(3, self.rand, 3))

    def step(self, a):
        return numpy.random.randint(low=10, high=40, size=(3, self.rand, 3)), 1, False, None

    def nActions(self):
        return 2

    def dState(self):
        return (3, self.rand, 3)

    def sampleAction(self):
        return numpy.random.randint(2)

    def prettifyState(self, rawState):
        # F, V, D = self.dState()
        b = len(rawState)
        return rawState.reshape(b, 3, -1, 3)

    def destroy(self):
        pass
