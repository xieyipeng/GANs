import torch.nn as nn
import Config

class discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        parame = [Config.ImageSize**2, 256, 1]

        self.dis = nn.Sequential(
            nn.Linear(parame[0], parame[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(parame[1], parame[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(parame[1], parame[2]),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        x = x.squeeze(-1)
        return x
