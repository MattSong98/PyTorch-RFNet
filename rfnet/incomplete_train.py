import os
import argparse
import logging
import numpy as np

import torch
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data import DataLoader, random_split

from rfnet.models.models import Modelv2
from rfnet.utils.utils import seed_setup, logging_setup, tensorboard_record, model_save
from rfnet.utils.utils import LR_Scheduler, loss_compute, AverageMeter
from rfnet.utils.datasets import Brats_loadall, init_fn, get_mask_combinations_exp1


def equally_spaced_list(length, num):
    q = length // num
    r = length % num
    ans = [q for i in range(num)]
    for j in range(r):
        ans[j] += 1
    return ans


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--patch_size', default=80, type=int)
    parser.add_argument('--num_sites', default=5, type=int)
    parser.add_argument('--datapath', default=None, type=str)
    parser.add_argument('--savepath', default=None, type=str)
    parser.add_argument('--resume', default=None, type=str)              
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--iter_per_epoch', default=150, type=int)
    parser.add_argument('--region_fusion_start_epoch', default=100, type=int)
    parser.add_argument('--seed', default=1024, type=int)
    args = parser.parse_args()
    return args


def train(train_loader, 
          model, 
          optimizer, 
          lr_scheduler, 
          num_cls, 
          num_epochs, 
          iter_per_epoch, 
          region_fusion_start_epoch, 
          writer, 
          ckpts):

    train_iter = iter(train_loader)

    train_avg_loss = AverageMeter()
    train_avg_loss_list = AverageMeter()
    
    for epoch in range(num_epochs): 

        model.train()
        model.module.is_training = True
        step_lr = lr_scheduler(optimizer, epoch)

        for i in range(iter_per_epoch): 

            step = (i + 1) + epoch * iter_per_epoch

            ### Loading Data ###
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)

            x, target, mask = data[:3]
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            ### Forward Propagation ###
            fuse_pred, sep_preds, prm_preds = model(x, mask)

            ### Computing Loss ###
            loss, loss_list = loss_compute(fuse_pred, sep_preds, prm_preds, target, num_cls, epoch, region_fusion_start_epoch)

            ### Updating Weights ###
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### Tensorboard Records (Iteration) ### 
            tensorboard_record(loss.item(), [l.item() for l in loss_list], writer, step)

            ### Logging (Iteration) ###
            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(loss_list[0].item(), loss_list[1].item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(loss_list[2].item(), loss_list[3].item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(loss_list[4].item(), loss_list[5].item())
            logging.info(msg)

            ### Updating Loss AverageMeters ###
            train_avg_loss.update(loss.item())
            train_avg_loss_list.update(np.array([l.item() for l in loss_list]))
        
        ### Tensorboard Records (Epoch) ###
        tensorboard_record(train_avg_loss.avg, train_avg_loss_list.avg.tolist(), writer, epoch+1, step_lr, prefix="epoch_")

        ### Logging (Epoch) ###
        msg = 'Avg Training Loss: Epoch {}/{}, Loss {:.4f}, '.format((epoch+1), num_epochs, train_avg_loss.avg)
        msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(train_avg_loss_list.avg[0], train_avg_loss_list.avg[1])
        msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(train_avg_loss_list.avg[2], train_avg_loss_list.avg[3])
        msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(train_avg_loss_list.avg[4], train_avg_loss_list.avg[5])
        logging.info(msg)

        ### Saving PyTorch model ###
        model_save(model, optimizer, ckpts, epoch, num_epochs)


def run():

    ############## Setup ###############

    # python cmd parser
    args = parse()

    # setup seeds
    seed_setup(args.seed)

    # setup logging
    logging_setup(args.savepath, lfile='train_log.txt', to_console=True)

    # setup checkpoint savepath
    ckpts = args.savepath

    # setup tensorboard writer
    writer = SummaryWriter(os.path.join(args.savepath, 'tensorboard'))

    # setup cudnn engine
    cudnn.benchmark = False 
    cudnn.deterministic = True

    ############## Dataset ##############

    train_file = 'train.txt'

    train_transforms = 'Compose([RandCrop3D(({},{},{})), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'.format(args.patch_size, args.patch_size, args.patch_size)

    num_cls = 4

    train_set = Brats_loadall(transforms=train_transforms, 
                              root=args.datapath, 
                              num_cls=num_cls, 
                              train_file=train_file,
                              mask_generator=get_mask_combinations_exp1)

    
    site_lengths = equally_spaced_list(len(train_set), args.num_sites)

    site_datasets = random_split(train_set, site_lengths)

    site_train_loaders = [DataLoader(dataset=site_dataset,
                                     batch_size=args.batch_size,
                                     num_workers=3,
                                     pin_memory=True,
                                     shuffle=True,
                                     worker_init_fn=init_fn)  
                                     for site_dataset in site_datasets]
    
    ########## Setting Models ###########

    models = [Modelv2(num_cls=num_cls) for i in range(args.num_sites)]
    models = [torch.nn.DataParallel(model).cuda() for model in models]

    ########## Scheduler & Optimizers ##########

    # one lr_scheduler is enough whereas each model requires an independant optimizer.
    lr_scheduler = LR_Scheduler(args.lr, args.num_epochs)
    optimizers = []
    for i in range(args.num_sites):
        train_params = [{'params': models[i].parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
        optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
        optimizers.append(optimizer)

    ########## Training #########
    
    train(site_train_loaders[0], 
          models[0], 
          optimizers[0], 
          lr_scheduler, 
          num_cls, 
          args.num_epochs, 
          args.iter_per_epoch, 
          args.region_fusion_start_epoch, 
          writer, 
          ckpts)


if __name__ == '__main__':
    run()
