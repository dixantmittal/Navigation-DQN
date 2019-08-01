import random
import threading
import time

import numpy

import Logger

logger = Logger.logger


class ReplayMemory(object):
    memory = []
    lock = threading.Lock()
    memoryEmpty = True

    def __init__(self, size):
        self.size = size
        self.position = 0
        self.buffer = []

        self.shutdownSync = False

        self.syncThread = threading.Thread(target=self.sync, name='ReplayMemorySync_{}'.format(id(self)))
        self.syncThread.start()
        logger.info('Started thread: Memory Sync')

    def sync(self):
        while not self.shutdownSync:
            ReplayMemory.lock.acquire()
            logger.debug('Syncing memory.')
            try:
                ReplayMemory.memory = ReplayMemory.memory + self.buffer
                ReplayMemory.memory = ReplayMemory.memory[:self.size]
                self.buffer = []
                logger.debug('Syncing finished.')
            finally:
                ReplayMemory.lock.release()

            if len(ReplayMemory.memory) > 0:
                ReplayMemory.memoryEmpty = False
            time.sleep(2)

        logger.info('Stopped thread: Memory Sync')

    def push(self, state, action, next_state, reward, terminate):

        # Combine the experience into 1 big array and store it on next position. Ordering is important
        ReplayMemory.lock.acquire()
        try:
            self.buffer.append(numpy.hstack((state.reshape(-1), action, next_state.reshape(-1), reward, terminate)))
        finally:
            ReplayMemory.lock.release()

    @staticmethod
    def sample(batch_size):
        # Change batch size if memory is not big enough
        batch_size = min(batch_size, len(ReplayMemory.memory))

        # Return a sampled batch
        return random.sample(ReplayMemory.memory, batch_size)

    def stop(self):
        self.shutdownSync = True
        self.syncThread.join()
