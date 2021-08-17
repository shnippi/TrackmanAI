import pathlib
from AE_networks import VAE_net, VAE_net_64, VanillaVAE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.utils import save_image
import numpy as np
import cv2
import time
from PIL import Image
from matplotlib import pyplot as plt
import gc

torch.cuda.empty_cache()
gc.collect()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("training")

batch_size = 16
learning_rate = 1e-3
num_epochs = 100
width = 64

train_mnist = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.Resize(64), transforms.ToTensor()]
    ),
)

test_mnist = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.Resize(64), transforms.ToTensor()]
    ),
)

train_loader = torch.utils.data.DataLoader(train_mnist, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_mnist, batch_size=batch_size, shuffle=True)

net = VanillaVAE().to(device)

model_file_name = "models/VAE_MNIST_100_vanilla.model"
net.load_state_dict(torch.load(model_file_name, map_location=device))
net = net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


net.train()
size = len(train_loader.dataset)

for epoch in range(num_epochs):
    total_loss = 0
    for batch, (x, y) in enumerate(train_loader):
        #
        # print(x.shape)
        # plt.imshow(x[0][0].to("cpu"), "gray")
        # plt.show()

        x, y = x.to(device), y.to(device)
        out, original, mu, logVar = net(x)
        loss, recon_loss, kld_loss = net.loss_function(out, original, mu, logVar,
                                                       M_N=batch_size / len(train_loader))

        total_loss += loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print('Epoch {}: Loss {}'.format(epoch, total_loss))

    results_dir = pathlib.Path("models")
    save_dir = results_dir / f"VAE_MNIST_100_vanilla.model"
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), save_dir)
