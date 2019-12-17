from model.train import train
from model.im2hi import IM2HI
from model.dataloader import trainloader, validloader
from torchvision import transforms
from model.helper.utils import Logger

# logger = Logger('im2hi', comment='test') # save important files each runs

COLAB = False
BATCH_SIZE = 64
AUGMENT = False

dataloader = {
    'train': trainloader(colab=COLAB, batch_size=BATCH_SIZE, augment=AUGMENT), 
    'val': validloader(colab=COLAB, batch_size=BATCH_SIZE)
}

net = IM2HI()
# print(net)
# print(net.parameters())
train(net, dataloader, learning_rate=1e-4, model_name='im2height')