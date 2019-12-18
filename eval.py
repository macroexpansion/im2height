from model.train import train
from model.im2hi import IM2HI
from model.dataloader import testloader
from model.evaluate import evaluate
import torch

COLAB = False
BATCH_SIZE = 64

test_loader = testloader(colab=COLAB, batch_size=BATCH_SIZE)

net = IM2HI()
net.load_state_dict(torch.load('weights/im2height.pt'))
evaluate(net, test_loader)