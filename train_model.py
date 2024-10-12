import torch
import torchvision.datasets as dsets
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import numpy as np
import random
from model_vae import VAE
from tqdm import tqdm_notebook, tqdm
import torchvision.models as models

batch_size = 32
image_size = 64
num_epochs = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed=7575):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=['3', '8', '15']):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.layers = layers
        _ = self.vgg.eval()

        for param in self.parameters():
            param.requires_grad = False  # Отключаем градиенты для VGG

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:  # Сохраняем фичи с указанных слоев
                features.append(x)
        return features


def get_data():
    # transform = transforms.Compose([
    #     transforms.Resize((image_size, image_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomRotation(degrees=10),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #     transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
    #     transforms.RandomGrayscale(p=0.1),
    #
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),

    ])

    dset = dsets.ImageFolder(root="/home/dm/Документы/some/hands_sber/data/",
                             transform=transform)
    # cut the size of the dataset
    dataset_train, dataset_val = torch.utils.data.random_split(dset,
                                                               [len(dset) - len(dset) // 6, len(dset) // 6],
                                                               generator=torch.Generator().manual_seed(42))
    dataloader = torch.utils.data.DataLoader(dataset_train,
                                             batch_size=batch_size,
                                             drop_last=True, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=batch_size,
                                                 drop_last=True, shuffle=True)

    return dataloader, val_dataloader


def load_model():
    model = VAE()
    model = model.to(device)

    try:
        model.load_state_dict(torch.load('weights/vae7.pth'))
    except:
        print("Weights not found ):")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, betas=(0.0, 0.999), weight_decay=0.0001)

    return model, optimizer


def vae_loss(recon_x, x, mu, logvar, perceptual_loss_fn) -> float:
    BCE = F.binary_cross_entropy(recon_x.view(-1, image_size * image_size * 3),
                                 x.view(-1, image_size * image_size * 3), reduction='sum')

    # recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # x_features = perceptual_loss_fn(x.to(device))
    # recon_x_features = perceptual_loss_fn(recon_x.to(device))
    # perceptual_loss = 0
    # for x_feat, recon_x_feat in zip(x_features, recon_x_features):
    #     perceptual_loss += F.mse_loss(recon_x_feat.to('cpu'), x_feat.to('cpu'), reduction='sum')
    # perceptual_loss = F.mse_loss(recon_x_feat.to('cpu'), x_feat.to('cpu'), reduction='sum')
    return BCE + KLD  # + perceptual_loss*0.2


def train_model(model, optimizer, dataloader, val_dataloader, perceptual_loss_fn):
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=4, verbose=True, eps=1e-6)
    for epoch in range(100):
        train_loss = 0
        model.train()
        for data, _ in tqdm(dataloader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data.to(device))

            # loss = vae_loss(recon_batch.cpu(), data.cpu(), mu.cpu(), logvar.cpu(), perceptual_loss_fn)
            loss = vae_loss(recon_batch.cpu(), data.cpu(), mu.cpu(), logvar.cpu(), perceptual_loss_fn)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        scheduler.step(train_loss)

        print('epoch %d, loss %.4f' % (epoch, train_loss / (len(dataloader) * batch_size)))
        torch.save(model.state_dict(), "weights/vae7.pth")

        if epoch % 5 == 0:
            model.eval()
            val_loss = 0
            for data, _ in tqdm(val_dataloader):
                recon_batch, mu, logvar = model(data.to(device))
                loss = vae_loss(recon_batch.cpu(), data.cpu(), mu.cpu(), logvar.cpu(), perceptual_loss_fn)
                val_loss += loss.item()

            print('epoch %d, loss %.4f' % (epoch, val_loss / (len(val_dataloader) * batch_size)))


if __name__ == '__main__':
    seed_everything()
    dataloader, val_dataloader = get_data()
    model, optimizer = load_model()
    perceptual_loss_fn = VGGPerceptualLoss().to(device)
    train_model(model, optimizer, dataloader, val_dataloader, perceptual_loss_fn)
