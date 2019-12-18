import torch
from model.train import train
from model.im2hi import IM2HI
from model.dataloader import trainloader, validloader
from torchvision import transforms
from model.helper.utils import Logger

comment = 'im2hi_loss'
logger = Logger('im2hi', comment=comment) # save important files each runs

COLAB = True
BATCH_SIZE = 16
AUGMENT = True

dataloader = {
    'train': trainloader(colab=COLAB, batch_size=BATCH_SIZE, augment=AUGMENT), 
    'val': validloader(colab=COLAB, batch_size=BATCH_SIZE)
}

net = IM2HI()
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
      num_epochs=200,
      model_name='im2height_loss', 
      comment=comment)


# unet1: valid 0.724712 test 0.6318