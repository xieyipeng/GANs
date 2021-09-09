import torch.nn as nn

class generator(nn.Module):
    def __init__(self):
        super().__init__()

        parame = [100, 256, 784]

        self.gen = nn.Sequential(
            nn.Linear(parame[0], parame[1]),
            nn.ReLU(True),
            nn.Linear(parame[1], parame[1]),
            nn.ReLU(True),
            nn.Linear(parame[1], parame[2]),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.gen(x)
        return x