from model.train import train
from model.im2hi import IM2HI

net = IM2HI()
train(net, learning_rate=1e-4, colab=True)