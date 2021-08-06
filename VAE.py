import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
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

batch_size = 10
learning_rate = 1e-3
num_epochs = 100
width = 250


def npy_loader(path):
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    sample = torch.from_numpy(np.load(path))
    np.load = np_load_old
    return sample


train_dataset = datasets.DatasetFolder(
    root='data_train',
    loader=npy_loader,
    extensions='.npy',
)

test_dataset = datasets.DatasetFolder(
    root='data_test',
    loader=npy_loader,
    extensions='.npy',
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=64 * 2 * 2, zDim=64):
        super(VAE, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=(2, 2), stride=2)

        self.index1 = None
        self.index2 = None
        self.index3 = None

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, 5, stride=2)
        self.encConv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.encConv3 = nn.Conv2d(32, 64, 5, stride=2)

        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)

        self.decConv1 = nn.ConvTranspose2d(64, 32, 5, stride=2, output_padding=(1, 1))
        self.decConv2 = nn.ConvTranspose2d(32, 16, 5, stride=2)
        self.decConv3 = nn.ConvTranspose2d(16, imgChannels, 5, stride=2, output_padding=(1, 1))

    def encoder(self, x):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        # print(x.shape)

        x, self.index1 = self.pool(self.encConv1(x))
        x = F.relu(x)
        # print(x.shape)

        x, self.index2 = self.pool(self.encConv2(x))
        x = F.relu(x)
        # print(x.shape)

        x, self.index3 = self.pool(self.encConv3(x))
        x = F.relu(x)
        # print(x.shape)

        x = x.view(-1, 64 * 2 * 2)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 64, 2, 2)

        x = F.relu(self.decConv1(self.unpool(x, self.index3, output_size=torch.Size([64, 32, 5, 5]))))
        # print(x.shape)
        x = F.relu(self.decConv2(self.unpool(x, self.index2, output_size=torch.Size([64, 32, 29, 29]))))
        # print(x.shape)
        x = torch.sigmoid(self.decConv3(self.unpool(x, self.index1, output_size=torch.Size([64, 32, 123, 123]))))
        # print(x.shape)
        return x

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar

    def get_z(self, x):
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)

        return z


net = VAE().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):
    for idx, data in enumerate(train_loader, 0):
        imgs, _ = data
        imgs = imgs.to(device)
        imgs = imgs.permute(0, 1, 4, 2, 3)  # switch from NHWC to NCHW
        imgs = transforms.Grayscale().forward(imgs)  # convert to grayscale
        # print(imgs.shape)

        for file in imgs:
            file = file / 256

            # print(file.shape)
            # plt.imshow(file[0][0].to("cpu"), "gray")
            # plt.show()

            # iterate over batch
            for batch in torch.split(file, batch_size):
                # print(batch.shape)

                # Feeding a batch of images into the network to obtain the output image, mu, and logVar
                out, mu, logVar = net(batch)

                # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
                kl_divergence = 0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
                loss = F.binary_cross_entropy(out, batch, reduction='sum') - kl_divergence

                # Backpropagation based on the loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    print('Epoch {}: Loss {}'.format(epoch, loss))

    results_dir = pathlib.Path("models")
    save_dir = results_dir / f"VAE.model"
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), save_dir)

"""
The following part takes a random image from test loader to feed into the VAE.
Both the original image and generated image from the distribution are shown.
"""

import matplotlib.pyplot as plt
import numpy as np
import random

net.eval()
with torch.no_grad():
    for data in random.sample(list(test_loader), 1):
        imgs, _ = data
        imgs = imgs.to(device)
        imgs = imgs.permute(0, 1, 4, 2, 3)  # switch from NHWC to NCHW
        imgs = transforms.Grayscale().forward(imgs)  # convert to grayscale
        imgs = imgs[0]
        imgs = imgs / 256
        # img = np.transpose(imgs[0].cpu().numpy(), [1, 2, 0])
        # plt.subplot(121)
        # plt.imshow(np.squeeze(img))
        plt.imshow(imgs[0][0].to("cpu"), "gray")
        plt.show()
        out, mu, logVAR = net(imgs)
        # outimg = np.transpose(out[0].cpu().numpy(), [1, 2, 0])
        # plt.subplot(122)
        # plt.imshow(np.squeeze(outimg))
        plt.imshow(out[0][0].to("cpu"), "gray")
        plt.show()
