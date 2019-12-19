import torch
import torch.nn as nn
import torch.optim as optim
from model.metric import ssim, SSIM
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def evaluate(net,testloader, criterion='',model_name='im2height'):
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    use_gpu = torch.cuda.is_available()
    device  = 'cuda:0' if use_gpu else 'cpu'
    if use_gpu:
        print('Using CUDA')
        # net.cuda()

    data_size = 500
    since = time.time()

    running_loss = 0.0
    running_l1 = 0.0
    running_mse = 0.0
    running_ssim = 0.0

    net.eval()
    i = 1
    for image, mask in tqdm(testloader):
        image = image.to(device)
        mask = mask.to(device)

        with torch.set_grad_enabled(False):
            output = net(image)
            _output, _image, _mask = output, image, mask
            ssim_value = ssim(output, mask)
            mask, output = get_nonzero_value(mask, output)
            if mask.size(0) == 0:
                del image, mask, output
                torch.cuda.empty_cache()
                continue
            loss = criterion(output, mask)
            l1_value = l1(output, mask)
            mse_value = mse(output, mask)
            save_fig(_image, _mask, _output, i)
            i += 1

        running_loss += loss.item() * image.size(0)
        running_ssim += ssim_value.item() * image.size(0)
        running_l1 +=  l1_value.item() * image.size(0)
        running_mse += mse_value.item() * image.size(0)

        del image, mask, output
        torch.cuda.empty_cache()

    epoch_loss = running_loss / i
    epoch_ssim = running_ssim / i
    epoch_l1 = running_l1 / i
    epoch_mse = running_mse / i

    print('{} -> Loss: {:.4f} SSIM: {:.4f} L1: {:.4f} MSE: {:.4f}'.format('Evaluate', epoch_loss, epoch_ssim, epoch_l1, epoch_mse))
    print('\ttime', time.time() - since)


def save_fig(img_, mask_, output_, i):
    img = img_.cpu()
    mask = mask_.cpu()
    output = output_.cpu()

    output = torch.squeeze(output, 0)
    output = torch.cat((output, output, output))

    img, mask = torch.squeeze(img, 0), torch.squeeze(mask, 0)
    mask = torch.cat((mask, mask, mask))

    fig, (a1, a2,a3) = plt.subplots(1, 3, figsize=(15,5))
    a1.axes.get_xaxis().set_ticks([])
    a1.axes.get_yaxis().set_ticks([])
    a2.axes.get_xaxis().set_ticks([])
    a2.axes.get_yaxis().set_ticks([])
    a3.axes.get_xaxis().set_ticks([])
    a3.axes.get_yaxis().set_ticks([])

    a1.set_xlabel('input')
    a2.set_xlabel('mask')
    a3.set_xlabel('output')

    a1.imshow((img.permute(1,2,0) * 255).type(torch.uint8))
    a2.imshow((mask.permute(1,2,0) * 255).type(torch.uint8))
    a3.imshow((output.permute(1,2,0) * 255).type(torch.uint8))
    plt.savefig('savefigs/save_{}'.format(i), bbox_inches='tight')
    plt.close(fig)
    # plt.show()


def get_nonzero_value(x, y) -> 'x, y 4-dimensions':
    indices = (x > 0).nonzero()
    indexed_1 = x[indices[:,0], indices[:,1], indices[:,2], indices[:,3]]
    indexed_2 = y[indices[:,0], indices[:,1], indices[:,2], indices[:,3]]
    return indexed_1, indexed_2