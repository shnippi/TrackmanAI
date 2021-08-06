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


dataset = datasets.DatasetFolder(
    root='data_train',
    loader=npy_loader,
    extensions='.npy',
)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


net = autoencoder().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

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
                output = net(batch)
                loss = criterion(output, batch)

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