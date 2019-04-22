import numpy as np
import torch

from constants import *
from replay_memory import ReplayMemory


class DQNTrainer(object):
    def __init__(self, simulator, network):
        # store the network and simulator
        self.simulator = simulator
        self.network = network

        # Replay buffer
        self.memory = ReplayMemory(10000)
        self.trauma = ReplayMemory(1000)

        # check if GPU acceleration is possible
        self.cuda_enabled = torch.cuda.is_available()

    def train(self, _batch_size = 64, _learning_rate = 1e-4, epoch = 1000, _target_update_frequency = 10, eps = 1, gamma = 0.9,
              verbose = False, _plot_frequency = 10):

        # initialise a test set
        print('Loading Test Set...')
        _test_set = []
        for i in range(10):
            _test_set.append(self.simulator.reset())
        print('Test Set Loaded!')

        _test_set = torch.cuda.FloatTensor(_test_set) if self.cuda_enabled else torch.FloatTensor(_test_set)

        # Initialise Metrics
        _episode_reward_trend = []
        _test_q_values_trend = []

        _policy_net = self.network
        _target_net = self.network.copy()

        # initialise optimiser and loss function
        optimiser = torch.optim.SGD(_policy_net.parameters(), lr = _learning_rate, weight_decay = 1e-4)
        _loss_fn = torch.nn.MSELoss()

        # move to cuda
        if self.cuda_enabled:
            _policy_net.cuda()
            _target_net.cuda()
            _loss_fn.cuda()

        len_states = IMG_HEIGHT * IMG_WIDTH * nCHANNELS * nFRAMES

        print('Starting Training...')
        for itr in range(epoch):
            # Copy Policy Network to Target Network
            if itr % _target_update_frequency == 0:
                _target_net = _policy_net.copy().cuda() if self.cuda_enabled else _policy_net.copy()

            # Decaying Epsilon (exploration factor)
            eps = max(0.1, eps ** itr)

            _terminate = False
            _episode_reward = 0

            # Reset simulator for new episode
            _state = self.simulator.reset()

            _episode_length = 0

            while not _terminate:
                # ACT
                if np.random.rand() > eps:
                    _state = torch.cuda.FloatTensor(_state) if self.cuda_enabled else torch.FloatTensor(_state)
                    _action = _policy_net(_state).argmax().item()
                    _state = _state.cpu().numpy()

                else:
                    _action = self.simulator.sample_action()

                # take action and get next state
                result = self.simulator.step(_action)
                _next_state = result['state']
                _reward = result['reward']
                _terminate = result['terminate']

                # store into experience memory
                self.memory.push(_state, _action, _next_state, _reward, int(not _terminate))

                # store collisions separately
                if _reward < 0:
                    self.trauma.push(_state, _action, _next_state, _reward, int(not _terminate))

                _state = _next_state
                _episode_reward += _reward * gamma ** _episode_length
                _episode_length += 1

                # OPTIMIZE POLICY
                # Priority sample a random batch of experiences. Sample collisions more frequently than normal experience
                if np.random.rand() < 0.1 and len(self.trauma) > 0:
                    batch = self.trauma.sample(_batch_size)
                else:
                    batch = self.memory.sample(_batch_size)

                # slice them to get state and actions
                batch = torch.cuda.FloatTensor(batch) if self.cuda_enabled else torch.FloatTensor(batch)
                state, action, next_state, reward, terminate = torch.split(batch, [len_states, nACTIONS, len_states, 1, 1], dim = 1)

                # reshape the data
                state = state.view(-1, nCHANNELS * nFRAMES, IMG_HEIGHT, IMG_WIDTH)
                next_state = next_state.view(-1, nCHANNELS * nFRAMES, IMG_HEIGHT, IMG_WIDTH)

                # find the target value
                _target = reward + terminate * gamma * _target_net(next_state).max(dim = 1)[0].unsqueeze(dim = 1)

                # Calculate Q value
                _policy = (_policy_net(state) * action).sum(dim = 1).unsqueeze(dim = 1)

                # find loss
                loss = _loss_fn(_policy, _target)

                # Backprop
                optimiser.zero_grad()
                loss.backward()

                # Clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad.clip_grad_value_(_policy_net.parameters(), 10)
                optimiser.step()

                # Terminate episode if contribution of next state is small.
                if gamma ** _episode_length < 0.1:
                    break

            # Store Evaluation Metrics
            _test_q_values_trend.append(_policy_net(_test_set).max(dim = 1)[0].mean().item())
            _episode_reward_trend.append(_episode_reward)

            # Print statistics
            if verbose:
                print('Epoch: ', itr)
                print('Episode Length: ', _episode_length)
                print('Episode Reward: ', _episode_reward_trend[-1])
                print('Test Q-Values', _test_q_values_trend[-1])
                print()

            # Checkpoint
            np.save('q_value_trend.npy', _test_q_values_trend)
            np.save('_episode_reward_trend.npy', _episode_reward_trend)
            _policy_net.save('policy_net.pth')
            _policy_net.cuda()
