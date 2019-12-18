import torch
import torch.nn as nn
import torch.optim as optim
from model.helper.utils import EarlyStopping
from model.dataloader import trainloader, validloader
import time
from tqdm import tqdm
from model.metric import ssim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, date


def train(net, dataloader, criterion=None, optimizer=None, num_epochs=100, model_name='im2height', learning_rate=1e-4, comment='comment'):
    use_gpu = torch.cuda.is_available()
    device  = 'cuda:0' if use_gpu else 'cpu'
    if use_gpu:
        print('Using CUDA')
        # net.cuda()
    
    train_size = 2376
    valid_size = 370

    since = time.time()
    now = date.today().strftime('%d-%m-%Y_') + datetime.now().strftime('%H:%M:%S')

    train_writer = SummaryWriter(log_dir='logs-tensorboard/%s/train' % now, comment='-'+comment)
    val_writer = SummaryWriter(log_dir='logs-tensorboard/%s/val' % now, comment='-'+comment)
    ssim_writer = SummaryWriter(log_dir='logs-tensorboard/%s/ssim' % now, comment='-'+comment)
    es = EarlyStopping(mode='max', patience=5)

    best_ssim = 0.0
    
    for epoch in range(num_epochs):
        start = time.time()
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            running_loss = 0.0
            running_ssim = 0.0
            
            m_loss, m_ssim = (0, 0)
            i = 0
            for image, mask in tqdm(dataloader[phase]):
                image = image.to(device)
                mask = mask.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = net(image)
                    loss = criterion(output, mask)
                    ssim_value = ssim(output, mask)
                    
                    # m_loss += loss.item()
                    # m_ssim += ssim_value.item()
                    if phase == 'train':
                        ssim_writer.add_scalar('loss %s' % str(epoch), loss.item(), i)
                        ssim_writer.add_scalar('ssim %s' % str(epoch), ssim_value.item(), i)
                        # m_loss, m_ssim = 0, 0
                
                    # if phase == 'train':
                        loss.backward()
                        optimizer.step()
                i += 1
                # print(image.size(0))
                running_loss += loss.item() * image.size(0)
                running_ssim += ssim_value.item() * image.size(0)

                del image, mask, output
                torch.cuda.empty_cache()

            data_size = train_size if phase == 'train' else valid_size
            epoch_loss = running_loss / data_size
            epoch_ssim = running_ssim / data_size

            print('{} -> Loss: {:.4f} SSIM: {:.4f}'.format(phase, epoch_loss, epoch_ssim))
            print('\ttime', time.time() - start)

            if phase == 'train':
                train_writer.add_scalar('L1Loss', epoch_loss, epoch)
                train_writer.add_scalar('SSIM', epoch_ssim, epoch)

            if phase == 'val':
                val_writer.add_scalar('L1Loss', epoch_loss, epoch)
                val_writer.add_scalar('SSIM', epoch_ssim, epoch)

                # ssim = epoch_ssim
                if es.step(epoch_ssim):
                    time_elapsed = time.time() - since
                    print('Early Stopping')
                    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                    print('Best val ssim: {:4f}'.format(best_ssim))
                    return

                if epoch_ssim > best_ssim:
                    best_ssim = epoch_ssim
                    print('Update best loss: {:4f}'.format(best_ssim))
                    torch.save(net.state_dict(), '{}.pt'.format(model_name))
                


if __name__ == '__main__':
    i1 = torch.rand((1,1,256,256))
    i2 = torch.rand((1,1,256,256))

    loss = ssim(i1,i1)
    print(loss.item())