import Logger
from simulator import *


class SimulatorFactory(object):
    @staticmethod
    def getInstance(className):
        Logger.logger.info('Instance requested for class %s', className)

        return globals()[className]()
