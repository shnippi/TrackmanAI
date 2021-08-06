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
from networks import AE_net, AE_net_no_pool, AE_skip

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

net = AE_net().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
# criterion = F.binary_cross_entropy(reduction="sum")

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
                # loss = F.binary_cross_entropy(output, batch, reduction='sum')

                # Backpropagation based on the loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    print('Epoch {}: Loss {}'.format(epoch, loss))

    results_dir = pathlib.Path("models")
    save_dir = results_dir / f"AE.model"
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), save_dir)