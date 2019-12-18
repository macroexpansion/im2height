import torch
import torch.nn as nn
import torch.optim as optim
from model.metric import ssim, SSIM
import time
from tqdm import tqdm

def evaluate(net,testloader, criterion='',model_name='im2height'):
    use_gpu = torch.cuda.is_available()
    device  = 'cuda:0' if use_gpu else 'cpu'
    if use_gpu:
        print('Using CUDA')
        # net.cuda()

    data_size = 500
    since = time.time()

    running_loss = 0.0
    running_ssim = 0.0

    net.eval()
    for image, mask in tqdm(testloader):
        image = image.to(device)
        mask = mask.to(device)

        with torch.set_grad_enabled(False):
            output = net(image)
            loss = criterion(output, mask)
            ssim_value = ssim(output, mask)

        running_loss += loss.item() * image.size(0)
        running_ssim += ssim_value.item() * image.size(0)

        del image, mask, output
        torch.cuda.empty_cache()

    epoch_loss = running_loss / data_size
    epoch_ssim = running_ssim / data_size

    print('{} -> Loss: {:.4f} SSIM: {:.4f}'.format('Evaluate', epoch_loss, epoch_ssim))
    print('\ttime', time.time() - since)