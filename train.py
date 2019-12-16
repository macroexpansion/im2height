from model.train import train
from model.im2hi import IM2HI
from model.dataloader import trainloader, validloader

colab = True
batch_size = 64
dataloader = {
    'train': trainloader(colab=colab, batch_size=batch_size), 
    'val': validloader(colab=colab, batch_size=batch_size)
}

net = IM2HI()
train(net, dataloader,learning_rate=1e-4, colab=colab)