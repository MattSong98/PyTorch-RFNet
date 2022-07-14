import argparse
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from rfnet.models.models import Model
from rfnet.utils.utils import logging_setup, seed_setup, AverageMeter, dice_compute, get_sliding_windows
from rfnet.utils.datasets import Brats_loadall_eval, get_mask_combinations


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--patch_size', default=80, type=int)
    parser.add_argument('--datapath', default=None, type=str)
    parser.add_argument('--savepath', default=None, type=str)
    parser.add_argument('--checkpoint', default=None, type=str)              
    parser.add_argument('--seed', default=1024, type=int)
    args = parser.parse_args()
    return args


def testing(test_loader, model, feature_mask, patch_size=80):

    model.eval()
    model.module.is_training=False

    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()

    num_cls = 4

    class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
    class_separate = 'ncr_net', 'edema', 'enhancing'

    for i, data in enumerate(test_loader):

        ##### Loading Data #####
        x = data[0].cuda()
        target = data[1].cuda()
        names = data[-1]

        mask = torch.from_numpy(np.array(feature_mask)) 
        mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        mask = mask.cuda()

        ###### Sliding Windows ######
        B, _, H, W, Z = x.size()
        h_idx_list, w_idx_list, z_idx_list = get_sliding_windows(H, W, Z, patch_size=patch_size)

        ###### Prediction ######
        one_tensor = torch.ones(1, 1, patch_size, patch_size, patch_size).float().cuda() 
        weight = torch.zeros(1, 1, H, W, Z).float().cuda()
        pred = torch.zeros(B, num_cls, H, W, Z).float().cuda()

        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size]
                    pred_part = model(x_input, mask)
                    pred[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += pred_part
                    weight[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += one_tensor

        weight = weight.repeat(B, num_cls, 1, 1, 1)
        pred = pred / weight
        pred = torch.argmax(pred, dim=1)

        ###### Computing Dice ######
        scores_separate, scores_evaluation = dice_compute(pred, target)

        ###### Logging (sample-i, mask-j) ######
        for k, name in enumerate(names):

            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            msg += '{:>20}, '.format(name)
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])
            logging.info(msg)

    ###### Logging (mask-j) #######
    msg = 'Average scores (mask):'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_separate.avg)])
    logging.info(msg)

    return vals_evaluation.avg


def run():

    ############## Setup ###############

    # python cmd parser
    args = parse()

    # setup seeds
    seed_setup(args.seed)

    # setup logging
    logging_setup(args.savepath, lfile='test_log.txt', to_console=True)

    # setup cudnn engine
    cudnn.benchmark = True 
    cudnn.deterministic = True

    ########## Datasets ###########

    test_file = 'test.txt'

    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

    num_cls = 4

    test_set = Brats_loadall_eval(transforms=test_transforms, 
                                  root=args.datapath, 
                                  test_file=test_file)

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True)     

    ########## Loading Models ###########

    model = Model(num_cls=num_cls) 
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    ########## Testing ##############

    masks, _, mask_name = get_mask_combinations()
    test_score = AverageMeter()

    with torch.no_grad():

        for i, mask in enumerate(masks):

            logging.info('mask | {}'.format(mask_name[i]))
            dice_score = testing(test_loader, model, feature_mask=mask, patch_size=args.patch_size)
            test_score.update(dice_score)

        logging.info('Avg scores: {}'.format(test_score.avg))


if __name__ == '__main__':
    run()
