import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import tqdm
import matplotlib.pyplot as plt

from model_utils import Discriminator, Generator
from matplotlib.pyplot import cm
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
style.use("ggplot")

def generate_real_samples(n):
    ''' Function: z = sin(5x) * cos(5y)/5 '''
    # Generate inputs between [0.2, 1.0001)
    X = np.random.uniform(0.2, 1.0001, n)
    y = np.random.uniform(0.2, 1.0001, n)
    z = np.sin(5*X) * (np.cos(5*y)/5)
    # Stacking the arrays
    X = X.reshape(n, 1)
    y = y.reshape(n, 1)
    z = z.reshape(n, 1)
    X = torch.from_numpy(np.hstack((X, y, z))).float()
    # Generating the real class labels
    y = torch.ones(n, 1) * 0.9
    return X, y

def generate_fake_samples(n):
    ''' 2-D function nonsense = nonsense + nonsense '''
    # Generating inputs between [-1, 1)
    X = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    z = np.random.uniform(-1, 1, n)
    # Stacking the arrays
    X = X.reshape(n, 1)
    y = y.reshape(n, 1)
    z = z.reshape(n, 1)
    X = torch.from_numpy(np.hstack((X, y, z))).float()
    # Generating the fake class labels
    y = torch.zeros(n, 1) * 0.1
    return X, y

def generate_latent_points(latent_dim, n):
    ''' Generate noise '''
    return torch.from_numpy(np.random.randn(latent_dim * n).reshape(n,latent_dim)).float()

#def performance_plot(model, latent_dim, n):
def performance_plot(model, latent_dim, n):
    ''' Plotting the generated points '''
    plt.clf()
    # Ground truth
    x_real, y_real = generate_real_samples(n)
    # Generated points
    with torch.no_grad():
        x_input = generate_latent_points(latent_dim, n).to(device)
        results = model(x_input).cpu().data.numpy()
    # Normalization colors
    norm_ax1 = plt.Normalize(x_real[:,2].min(), x_real[:,2].max())
    norm_ax2 = plt.Normalize(results[:,2].min(), results[:,2].max())
    colors_ax1 = cm.viridis(norm_ax1(x_real[:,2]))
    colors_ax2 = cm.viridis(norm_ax2(results[:,2]))
    
    # Plotting results
    ax = plt.figure()
    ax1 = ax.add_subplot(1, 2, 1, projection='3d')
    #ax1.plot_trisurf(x_real[:,0], x_real[:,1], x_real[:,2], facecolors=colors, linewidth=0, antialiased=False)
    ax1.scatter(x_real[:,0], x_real[:,1], x_real[:,2], facecolors=colors_ax1, linewidth=0, antialiased=False)
    ax2 = ax.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(results[:,0], results[:,1], results[:,2], facecolors=colors_ax2, linewidth=0, antialiased=False)
    plt.title('Performance plot')
    plt.show()

# Testing discriminator
def test_discriminator(n=3):
    ''' Testing the discriminator '''
    netD.eval()
    test_x, test_y = generate_real_samples(n)
    test_x_f, test_y_f = generate_fake_samples(n)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    test_x_f = test_x_f.to(device)
    test_y_f = test_y_f.to(device)
    with torch.no_grad():
        tmp = netD(test_x).reshape(-1)
        print(tmp)
        tmp = netD(test_x_f).reshape(-1)
        print(tmp)
        
# Hyperparameters
EPOCHS = 100000
lr = 0.0001
batch_size = 128
input_dim = 3
latent_dim = 5

# Choosing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}.')

# Creating the Discriminator and Generator
netD = Discriminator(input_dim).to(device)
netG = Generator(latent_dim).to(device)

# Setting the optimizers
optimD = optim.Adam(netD.parameters(), lr=lr)
optimG = optim.Adam(netG.parameters(), lr=lr)

# Setting the loss function
criterion = nn.BCELoss().to(device)

# Training
training = False 
if training:
    netD.train()
    netG.train()
    half_batch = batch_size // 2
    for epoch in range(EPOCHS):
        # Training discriminator with real samples
        X_real, y_real = generate_real_samples(half_batch)
        X_real = X_real.to(device)
        y_real = y_real.to(device)
        output = netD(X_real).reshape(-1)
        lossD_real = criterion(output, y_real)
        D_x = output.mean().item() # mean confidence
        
        # Training discriminator with fake samples
        noise = generate_latent_points(latent_dim, half_batch).to(device)
        X_fake = netG(noise)
        y_fake = (torch.zeros(half_batch)*0.1).to(device)
        output = netD(X_fake.detach()).reshape(-1)
        lossD_fake = criterion(output, y_fake)
        
        # Back prop on discriminator
        lossD = lossD_real + lossD_fake
        optimD.zero_grad()
        lossD.backward()
        optimD.step()
        
        # Training generator
        label = (torch.ones(half_batch)).to(device)
        output = netD(X_fake).reshape(-1)
        lossG = criterion(output, label)
        
        # Back prop on generator
        optimG.zero_grad()
        lossG.backward()
        optimG.step()
        print(f'Epoch: [{epoch}/{EPOCHS}], D_Loss: {lossD}, G_Loss: {lossG}, D_x: {D_x}')

#torch.save(netG, 'models/generator_100k')
netG = torch.load('models/generator_100k')
performance_plot(netG, latent_dim, 2000)