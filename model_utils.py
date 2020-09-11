import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim=3):
        super(Discriminator, self).__init__()
        # Initializing hidden layers
        self.linear1 = nn.Linear(input_dim, 25)
        nn.init.kaiming_uniform_(self.linear1.weight)
        self.linear2 = nn.Linear(25, 45)
        nn.init.kaiming_uniform_(self.linear2.weight)
        # Declaring discriminator net
        self.net = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.ReLU(),
            nn.Linear(45, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)
    
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        # Initializing hidden layers
        self.linear1 = nn.Linear(latent_dim, 15)
        nn.init.kaiming_uniform_(self.linear1.weight)
        self.linear2 = nn.Linear(15, 25)
        nn.init.kaiming_uniform_(self.linear2.weight)
        # Declaring generator net
        self.net = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.ReLU(),
            nn.Linear(25, 3),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)