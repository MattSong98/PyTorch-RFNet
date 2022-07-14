import os
import torch
import random
import logging
import numpy as np


class LR_Scheduler(object):
    def __init__(self, base_lr, num_epochs, mode='poly'):
        self.mode = mode
        self.lr = base_lr
        self.num_epochs = num_epochs

    def __call__(self, optimizer, epoch):
        if self.mode == 'poly':
            now_lr = round(self.lr * np.power(1 - np.float32(epoch)/np.float32(self.num_epochs), 0.9), 8) 
        self._adjust_learning_rate(optimizer, now_lr)
        return now_lr

    def _adjust_learning_rate(self, optimizer, lr):
        optimizer.param_groups[0]['lr'] = lr


def dice_loss(output, target, num_cls=4, eps=1e-7):
    target = target.float()
    for i in range(num_cls):
        num = torch.sum(output[:,i,:,:,:] * target[:,i,:,:,:])
        l = torch.sum(output[:,i,:,:,:])
        r = torch.sum(target[:,i,:,:,:])
        if i == 0:
            dice = 2.0 * num / (l+r+eps)
        else:
            dice += 2.0 * num / (l+r+eps)
    return 1.0 - 1.0 * dice / num_cls


def softmax_weighted_loss(output, target, num_cls=4):
    target = target.float()
    B, _, H, W, Z = output.size()
    for i in range(num_cls):
        outputi = output[:, i, :, :, :] 
        targeti = target[:, i, :, :, :] 
        weighted = 1.0 - (torch.sum(targeti, (1,2,3)) * 1.0 / torch.sum(target, (1,2,3,4)))
        weighted = torch.reshape(weighted, (-1,1,1,1)).repeat(1,H,W,Z)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss) 
    return cross_loss


def loss_compute(fuse_pred, sep_preds, prm_preds, target, num_cls, epoch, region_fusion_start_epoch):
    
    fuse_cross_loss = softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
    fuse_dice_loss = dice_loss(fuse_pred, target, num_cls=num_cls)
    fuse_loss = fuse_cross_loss + fuse_dice_loss

    sep_cross_loss = torch.zeros(1).cuda().float()
    sep_dice_loss = torch.zeros(1).cuda().float()
    for sep_pred in sep_preds:
        sep_cross_loss += softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
        sep_dice_loss += dice_loss(sep_pred, target, num_cls=num_cls)
    sep_loss = sep_cross_loss + sep_dice_loss

    prm_cross_loss = torch.zeros(1).cuda().float()
    prm_dice_loss = torch.zeros(1).cuda().float()
    for prm_pred in prm_preds:
        prm_cross_loss += softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
        prm_dice_loss += dice_loss(prm_pred, target, num_cls=num_cls)
    prm_loss = prm_cross_loss + prm_dice_loss

    if epoch < region_fusion_start_epoch:
        loss = fuse_loss * 0.0 + sep_loss + prm_loss
    else:
        loss = fuse_loss + sep_loss + prm_loss
    return loss, (fuse_cross_loss, fuse_dice_loss, sep_cross_loss, sep_dice_loss, prm_cross_loss, prm_dice_loss)


def model_save(model, optimizer, ckpts, epoch, num_epochs):

    file_name = os.path.join(ckpts, 'model_last.pth')
    torch.save({
        'epoch': epoch+1,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
        },
        file_name)
        
    if (epoch+1) % 50 == 0 or (epoch>=(num_epochs-10)):
        file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch+1))
        torch.save({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            },
            file_name)  


def logging_setup(savepath, lfile='log', to_console=False):

    ldir = savepath
    if not os.path.exists(ldir):
        os.makedirs(ldir)
    lfile = os.path.join(ldir, lfile)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename=lfile)

    if to_console:
      console = logging.StreamHandler()
      console.setLevel(logging.INFO)
      console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
      logging.getLogger('').addHandler(console)


def seed_setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def tensorboard_record(loss, loss_list, writer, step, step_lr=None, prefix=''):
    writer.add_scalar(prefix + 'loss', loss, global_step=step)
    writer.add_scalar(prefix + 'fuse_cross_loss', loss_list[0], global_step=step)
    writer.add_scalar(prefix + 'fuse_dice_loss', loss_list[1], global_step=step)
    writer.add_scalar(prefix + 'sep_cross_loss', loss_list[2], global_step=step)
    writer.add_scalar(prefix + 'sep_dice_loss', loss_list[3], global_step=step)
    writer.add_scalar(prefix + 'prm_cross_loss', loss_list[4], global_step=step)
    writer.add_scalar(prefix + 'prm_dice_loss', loss_list[5], global_step=step)
    if step_lr is not None:
        writer.add_scalar('lr', step_lr, global_step=step)


def dice(sep_output, sep_target, eps=1e-8):
    intersect = torch.sum(2 * (sep_output * sep_target), dim=(1,2,3)) + eps
    denominator = torch.sum(sep_output, dim=(1,2,3)) + torch.sum(sep_target, dim=(1,2,3)) + eps
    return intersect / denominator
    

def get_dice_score(output, target, labels):
    sep_outputs = torch.zeros_like(output).float()
    sep_targets = torch.zeros_like(output).float()
    for label in labels:
        sep_outputs += (output == label).float()
        sep_targets += (target == label).float()
    return dice(sep_outputs, sep_targets)
    

def post_et_dice(output, target, et_label=3):
    et_output = (output == et_label).float()
    et_target = (target == et_label).float()
    if torch.sum(et_output) < 500:
       sep_output = et_output * 0.0
    else:
       sep_output = et_output
    sep_target = et_target
    return dice(sep_output, sep_target)


def dice_compute(output, target):

    ncr_net_dice = get_dice_score(output, target, labels=[1])
    edema_dice = get_dice_score(output, target, labels=[2])
    enhancing_dice = get_dice_score(output, target, labels=[3])
    enhancing_dice_postpro = post_et_dice(output, target, et_label=3)
    dice_whole = get_dice_score(output, target, labels=[1, 2, 3])
    dice_core = get_dice_score(output, target, labels=[1, 3])

    dice_separate = torch.cat((torch.unsqueeze(ncr_net_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()


def get_sliding_windows(H, W, Z, patch_size=80):

    h_cnt = np.int(np.ceil((H - patch_size) / (patch_size * 0.5))) 
    h_idx_list = range(0, h_cnt)
    h_idx_list = [h_idx * np.int(patch_size * 0.5) for h_idx in h_idx_list]
    h_idx_list.append(H - patch_size)

    w_cnt = np.int(np.ceil((W - patch_size) / (patch_size * 0.5)))
    w_idx_list = range(0, w_cnt)
    w_idx_list = [w_idx * np.int(patch_size * 0.5) for w_idx in w_idx_list]
    w_idx_list.append(W - patch_size)

    z_cnt = np.int(np.ceil((Z - patch_size) / (patch_size * 0.5)))
    z_idx_list = range(0, z_cnt)
    z_idx_list = [z_idx * np.int(patch_size * 0.5) for z_idx in z_idx_list]
    z_idx_list.append(Z - patch_size)

    return h_idx_list, w_idx_list, z_idx_list


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count