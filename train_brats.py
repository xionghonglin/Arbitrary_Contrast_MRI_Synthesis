import argparse
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import datasets
import models
import utils
from eval import eval_psnr
from ortho_loss import OrthoLoss
from align_loss import AlignLoss
# import matplotlib.pyplot as plt



def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    log('{} dataset: size={}'.format(tag, len(dataset)))
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
   
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer):

    model.train()
    loss_im = nn.L1Loss()
    loss_ortho = OrthoLoss()
    loss_align = AlignLoss()

    loss0 = utils.Averager()
    loss1 = utils.Averager()
    loss2 = utils.Averager()
    loss3 = utils.Averager()
    loss_c = utils.Averager()
   
    
    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

     
        m_src = utils.random_selection([0,1,2,3])
        preds, contents = model([batch['0'],batch['1'],batch['2'],batch['3']],m_src)
        

        loss_0 = loss_im(preds[0], batch['0'])
        loss_1 = loss_im(preds[1], batch['1'])
        loss_2 = loss_im(preds[2], batch['2'])
        loss_3 = loss_im(preds[3], batch['3'])
        

        loss_img = (loss_0 + loss_1 + loss_2 + loss_3)/4 
        loss_content = (loss_align(contents[0], contents[1]) + loss_align(contents[0], contents[2]) + loss_align(contents[2], contents[1]))/3
        loss_o = (loss_ortho(contents[0]) + loss_ortho(contents[1]) + loss_ortho(contents[2]) + loss_ortho(contents[3]))/4

        loss = 0.8*loss_img  + 0.1*loss_content + loss_o*0.1
        loss0.add(loss_0.item())
        loss1.add(loss_1.item())
        loss2.add(loss_2.item())
        loss3.add(loss_3.item())
        loss_c.add(loss_content.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
               
    return loss_0.item(), loss_1.item(), loss_2.item(), loss_3.item(), loss_o.item()

def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()
    
    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        log_info.append('lr={:.4f}'.format(optimizer.param_groups[0]['lr']))

        train_loss = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        writer.add_scalars('loss0', {'train': train_loss[0]}, epoch)
        log_info.append('loss0={:.4f}'.format(train_loss[0]))
        writer.add_scalars('loss1', {'train': train_loss[1]}, epoch)
        log_info.append('loss1={:.4f}'.format(train_loss[1]))
        writer.add_scalars('loss2', {'train': train_loss[2]}, epoch)
        log_info.append('loss2={:.4f}'.format(train_loss[2]))
        writer.add_scalars('loss3', {'train': train_loss[3]}, epoch)
        log_info.append('loss3={:.4f}'.format(train_loss[3]))
        writer.add_scalars('losscc', {'train': train_loss[4]}, epoch)
        log_info.append('losscc={:.4f}'.format(train_loss[4]))
    
        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res0, val_res1,val_res2,val_res3 = eval_psnr(val_loader, model_,
                
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'))
            
            log_info.append('val: psnr0={:.4f}'.format(val_res0))
            log_info.append('val: psnr1={:.4f}'.format(val_res1))
            log_info.append('val: psnr2={:.4f}'.format(val_res2))
            log_info.append('val: psnr3={:.4f}'.format(val_res3))

            writer.add_scalars('psnr0', {'val': val_res0}, epoch)
            writer.add_scalars('psnr1', {'val': val_res1}, epoch)
            writer.add_scalars('psnr2', {'val': val_res2}, epoch)
            writer.add_scalars('psnr3', {'val': val_res3}, epoch)
 
            if val_res0+val_res1+val_res2+val_res3 > max_val_v:
                max_val_v = val_res0+val_res1+val_res2+val_res3
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_brats.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
