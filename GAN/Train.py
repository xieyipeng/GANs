from Discriminator import discriminator
from Generator import generator
from torchvision import transforms, datasets
from torch.autograd import Variable
from torchvision.utils import save_image
import wandb

import torch.nn as nn
import Config
import torch
import os

wandb.init(project='GAN')

# create generator dir
if not os.path.exists(Config.GenerateResults):
    os.mkdir(Config.GenerateResults)

if not os.path.exists(Config.ModelResult):
    os.mkdir(Config.ModelResult)

# img pre deal
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = datasets.MNIST(root=Config.DataSet, train=True,
                       transform=trans, download=True)

dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=Config.BatchSize, shuffle=True)


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


D = discriminator()
G = generator()

if torch.cuda.is_available():
    print('use cuda')
    D = D.cuda()
    G = G.cuda()
else:
    print('cuda is not available!')

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=Config.lr)
g_optimizer = torch.optim.Adam(G.parameters(), lr=Config.lr)


for epoch in range(Config.Epoch):
    for i, (img, _) in enumerate(dataloader):

        # prepare
        z = Variable(torch.randn(img.size(0), Config.NoiseDim)).cuda()
        img = img.view(img.size(0), -1)
        real_img = Variable(img).cuda()

        real_label = Variable(torch.ones(img.size(0))).cuda()
        fake_label = Variable(torch.zeros(img.size(0))).cuda()

        # train D
        real_out = D(real_img)
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out

        fake_out = D(G(z).detach())
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out

        d_loss = d_loss_fake + d_loss_real
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # train G
        g_loss = criterion(D(G(z)), real_label)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # logging
        if (i + 1) % 100 == 0:
            wandb.log({
                "d_loss": d_loss.data.item(),
                "g_loss": g_loss.data.item(),
                "real scores": real_scores.data.mean(),
                "fake scores": fake_scores.data.mean(),
            })
            print('Epoch[{}/{}], d_loss:{:.6f}, g_loss:{:.6f}, real scores: {:.6f}, fake scores: {:.6f}'.format(
                epoch, Config.Epoch, d_loss.data.item(), g_loss.data.item(), real_scores.data.mean(), fake_scores.data.mean()))

        if epoch == 0:
            real_images = to_img(real_img.cpu().data)
            save_image(real_images, Config.GenerateResults +
                       '/real_images.png')

    # fake_images = to_img(fake_img.cpu().data)
    # save_image(fake_images, Config.GenerateResults + '/fake_images_{}.png'.format(epoch + 1))

    fake_images = to_img(G(z).cpu().data)
    wandb.log({"image_view": [wandb.Image(
        fake_images[i], caption="image{}".format(i + 1)) for i in range(16)]})

    torch.save(G.state_dict(), Config.ModelResult +
               '/generator_{}.pth'.format(epoch))
    torch.save(D.state_dict(), Config.ModelResult +
               '/discriminator_{}.pth'.format(epoch))
