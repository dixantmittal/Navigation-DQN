class Experience(object):
    def __init__(self, state, action, nextState, reward, terminate):
        self.state, self.action, self.nextState, self.reward, self.terminate = state, action, nextState, reward, terminate

    def get(self):
        return self.state, self.action, self.nextState, self.reward, self.terminate
