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
import wandb

torch.cuda.empty_cache()
gc.collect()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wandb.login()

print("training")

hyperparameters = dict(
    batch_size=16,
    learning_rate=1e-3,
    num_epochs=100,
    width=64,
    beta=16 / 100000,
)


def npy_loader(path):
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    sample = torch.from_numpy(np.load(path))
    np.load = np_load_old
    return sample


def make(config):
    train_dataset = datasets.DatasetFolder(
        root='data/data_train_64_game',
        loader=npy_loader,
        extensions='.npy',
    )

    # test_dataset = datasets.DatasetFolder(
    #     root='data/data_test_64_video',
    #     loader=npy_loader,
    #     extensions='.npy',
    # )

    test_dataset = train_dataset

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    net = VanillaVAE().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    return net, train_loader, test_loader, optimizer


def train(net, train_loader, optimizer, config):
    net.train()
    wandb.watch(net, log="all", log_freq=1)
    # training loop
    for epoch in range(config.num_epochs):
        total_loss = 0
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
                for batch in torch.split(file, config.batch_size):
                    # print(batch.shape)

                    # Feeding a batch of images into the network to obtain the output image, mu, and logVar

                    # out, mu, logVar = net(batch)
                    out, original, mu, logVar = net(batch)  # Vanilla_VAE

                    loss, recon_loss, kld_loss = net.loss_function(out, original, mu, logVar,
                                                                   M_N=config.beta)  # Vanilla_VAE

                    total_loss += loss

                    # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
                    # kl_divergence = 0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
                    # loss = F.binary_cross_entropy(out, batch, reduction='sum') - kl_divergence

                    # Backpropagation based on the loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        wandb.log({"epoch": epoch, "loss": total_loss})
        print('Epoch {}: Loss {}'.format(epoch, total_loss))

        results_dir = pathlib.Path("models")
        save_dir = results_dir / f"VAE.model"
        results_dir.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), save_dir)


with wandb.init(project="Trackmania", config=hyperparameters):
    config = wandb.config
    net, train_loader, test_loader, optimizer = make(config)
    train(net, train_loader, optimizer, config)

#
# import matplotlib.pyplot as plt
# import numpy as np
# import random

# net.eval()
# with torch.no_grad():
#     for data in random.sample(list(test_loader), 1):
#         imgs, _ = data
#         imgs = imgs.to(device)
#         imgs = imgs.permute(0, 1, 4, 2, 3)  # switch from NHWC to NCHW
#         imgs = transforms.Grayscale().forward(imgs)  # convert to grayscale
#         imgs = imgs[0]
#         imgs = imgs / 256
#         # img = np.transpose(imgs[0].cpu().numpy(), [1, 2, 0])
#         # plt.subplot(121)
#         # plt.imshow(np.squeeze(img))
#         plt.imshow(imgs[0][0].to("cpu"), "gray")
#         plt.show()
#         out, mu, logVAR = net(imgs)
#         # outimg = np.transpose(out[0].cpu().numpy(), [1, 2, 0])
#         # plt.subplot(122)
#         # plt.imshow(np.squeeze(outimg))
#         plt.imshow(out[0][0].to("cpu"), "gray")
#         plt.show()
