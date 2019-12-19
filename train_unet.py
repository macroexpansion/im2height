import torch
from model.train import train
from model.im2hi import IM2HI
from model.unet import UNet
from model.dataloader import trainloader, validloader
from torchvision import transforms
from model.helper.utils import Logger

comment = 'unet_augment'
logger = Logger('unet', comment=comment) # save important files each runs

COLAB = False
BATCH_SIZE = 1
AUGMENT = True
LR = 1e-5

dataloader = {
    'train': trainloader(colab=COLAB, batch_size=BATCH_SIZE, augment=AUGMENT), 
    'val': validloader(colab=COLAB, batch_size=BATCH_SIZE)
}

net = UNet(3, 1)
if torch.cuda.is_available(): 
    net.cuda()

criterion = torch.nn.L1Loss()
# optimizer = optim.SGD(net.parameters(), 
                        # lr=LR, 
                        # momentum=0.9, 
                        # nesterov=True, 
                        # weight_decay=1e-1)
optimizer = torch.optim.Adam(net.parameters(), 
                       lr=2e-5, 
                       weight_decay=1e-2)

train(net, dataloader, 
      criterion=criterion,
      optimizer=optimizer,
      num_epochs=100,
      model_name='unet_augment', 
      comment=comment)