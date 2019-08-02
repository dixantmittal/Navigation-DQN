import torch
import torch.nn as nn

import Logger


def fill(t, i, j, a, b):
    for m in range(a - 1, a + 2):
        for n in range(b - 1, b + 2):
            t[i, j, m, n] = 0.5
    t[i, j, a, b] = 1
    return t


# Custom layer to flatten the output in 1-D vector
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class QNetwork(nn.Module):
    def __init__(self, inDims, outDims):
        super(QNetwork, self).__init__()

        self.inDims = inDims
        self.outDims = outDims

        C, H, W = inDims

        self.net = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Flatten(),

            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=outDims)

        )

    def forward(self, x):
        t = torch.zeros(10, 3, 50, 101).to(x.get_device())
        for i, b in enumerate(x):
            for j, b_ in enumerate(b):
                t = fill(t, i, j, 48, 50)
                for b__ in b_:
                    t = fill(t, i, j, b__[1], b__[0])

        return self.net(t)

    def save(self, filename):
        # Switch network to CPU before saving to avoid issues.
        Logger.logger.debug('Saving network to %s', filename)
        torch.save(self.cpu().state_dict(), filename)

    def load(self, filename):
        # Load state dictionary from saved file
        Logger.logger.debug('Loading network from %s', filename)
        self.load_state_dict(torch.load(filename, map_location='cpu'))

    def copy(self, freeze=True):
        # Create a copy of self
        copied = QNetwork(self.inDims, self.outDims)
        copied.load_state_dict(self.state_dict())

        # Freeze its parameters
        if freeze:
            for params in copied.parameters():
                params.requires_grad = False

        return copied
