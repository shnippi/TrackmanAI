import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import cv2


class VAE_net(nn.Module):
    def __init__(self, imgChannels=1, featureDim=64 * 2 * 2, zDim=64):
        super(VAE_net, self).__init__()

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


class AE_net(nn.Module):
    def __init__(self, imgChannels=1, featureDim=64 * 2 * 2, zDim=64):
        super(AE_net, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=(2, 2), stride=2)

        self.index1 = None
        self.index2 = None
        self.index3 = None

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, 5, stride=2)
        self.encConv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.encConv3 = nn.Conv2d(32, 64, 5, stride=2)

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

        z = x.view(-1, 64 * 2 * 2)
        return z

    def decoder(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = z.view(-1, 64, 2, 2)

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
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def get_z(self, x):
        z = self.encoder(x)
        return z


class AE_net_no_pool(nn.Module):
    def __init__(self, imgChannels=1, featureDim=64 * 2 * 2, zDim=64):
        super(AE_net_no_pool, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 4, 2, stride=2)
        self.encConv2 = nn.Conv2d(4, 8, 2, stride=3)
        self.encConv3 = nn.Conv2d(8, 16, 2, stride=2)
        self.encConv4 = nn.Conv2d(16, 32, 3, stride=2)
        self.encConv5 = nn.Conv2d(32, 64, 2, stride=2)
        self.encConv6 = nn.Conv2d(64, 64, 2, stride=3)

        self.decConv1 = nn.ConvTranspose2d(64, 64, 2, stride=3)
        self.decConv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decConv3 = nn.ConvTranspose2d(32, 16, 2, stride=2, output_padding=(1, 1))
        self.decConv4 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.decConv5 = nn.ConvTranspose2d(8, 4, 2, stride=3)
        self.decConv6 = nn.ConvTranspose2d(4, imgChannels, 2, stride=2)

    def encoder(self, x):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        # print(x.shape)

        x = self.encConv1(x)
        x = F.relu(x)
        # print(x.shape)

        x = self.encConv2(x)
        x = F.relu(x)
        # print(x.shape)

        x = self.encConv3(x)
        x = F.relu(x)
        # print(x.shape)

        x = self.encConv4(x)
        x = F.relu(x)
        # print(x.shape)

        x = self.encConv5(x)
        x = F.relu(x)
        # print(x.shape)

        x = self.encConv6(x)
        x = F.relu(x)
        # print(x.shape)

        return x

    def decoder(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        # print(z.shape)
        x = F.relu(self.decConv1(z))
        # print(x.shape)
        x = F.relu(self.decConv2(x))
        # print(x.shape)
        x = F.relu(self.decConv3(x))
        # print(x.shape)
        x = F.relu(self.decConv4(x))
        # print(x.shape)
        x = F.relu(self.decConv5(x))
        # print(x.shape)
        x = torch.sigmoid(self.decConv6(x))
        # print(x.shape)
        return x

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def get_z(self, x):
        x = self.encoder(x)
        z = x.view(-1, 64 * 2 * 2)
        return z


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

class AE_skip(nn.Module):
    def __init__(self):
        super(AE_skip, self).__init__()
        self.leaky_reLU = nn.LeakyReLU(0.2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=1)
        self.softmax = nn.Softmax2d()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # encoder
        x = self.conv1(x)
        out1 = self.leaky_reLU(x)
        x = out1
        size1 = x.size()
        x, indices1 = self.pool(x)

        x = self.conv2(x)
        out2 = self.leaky_reLU(x)
        x = out2
        size2 = x.size()
        x, indices2 = self.pool(x)

        x = self.conv3(x)
        out3 = self.leaky_reLU(x)
        x = out3
        size3 = x.size()
        x, indices3 = self.pool(x)

        x = self.conv4(x)
        out4 = self.leaky_reLU(x)
        x = out4
        size4 = x.size()
        x, indices4 = self.pool(x)

        ######################
        x = self.conv5(x)
        x = self.leaky_reLU(x)

        x = self.conv6(x)
        x = self.leaky_reLU(x)
        ######################

        # decoder
        x = self.unpool(x, indices4, output_size=size4)
        x = self.conv7(torch.cat((x, out4), 1))
        x = self.leaky_reLU(x)

        x = self.unpool(x, indices3, output_size=size3)
        x = self.conv8(torch.cat((x, out3), 1))
        x = self.leaky_reLU(x)

        x = self.unpool(x, indices2, output_size=size2)
        x = self.conv9(torch.cat((x, out2), 1))
        x = self.leaky_reLU(x)

        x = self.unpool(x, indices1, output_size=size1)
        x = self.conv10(torch.cat((x, out1), 1))
        x = self.softmax(x)

        return x
