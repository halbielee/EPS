import os
import torch
import argparse
from torch.backends import cudnn

from util import pyutils

from module.dataloader import get_dataloader
from module.model import get_model
from module.optimizer import get_optimizer
from module.train import train_cls, train_eps

cudnn.enabled = True
torch.backends.cudnn.benchmark = False

_NUM_CLASSES = {'voc12': 20, 'coco': 80}

def get_arguments():
    parser = argparse.ArgumentParser()
    # session
    parser.add_argument("--session", default="eps", type=str)

    # data
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--dataset", default='voc12', type=str)
    parser.add_argument("--saliency_root", type=str)
    parser.add_argument("--train_list", default="metadata/voc12/train_aug.txt", type=str)
    parser.add_argument("--save_root", default='/log')

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--resize_size", default=(256, 512), type=int, nargs='*')

    # network
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--weights", required=True, type=str,
                        default='pretrained/ilsvrc-cls_rna-a1_cls1000_ep-0001.params')

    # optimizer
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--max_iters", default=10000, type=int)

    # hyper-parameters for EPS
    parser.add_argument("--tau", default=0.5, type=float)
    parser.add_argument("--lam", default=0.5, type=float)

    args = parser.parse_args()

    args.num_classes = _NUM_CLASSES[args.dataset]

    if 'cls' in args.network:
        args.network_type = 'cls'
    elif 'eps' in args.network:
        args.network_type = 'eps'
    else:
        raise Exception('No appropriate model type')

    return args


if __name__ == '__main__':

    # get arguments
    args = get_arguments()

    # set log
    args.log_folder = os.path.join(args.save_root, args.session)
    os.makedirs(args.log_folder, exist_ok=True)

    pyutils.Logger(os.path.join(args.log_folder, 'log_cls.log'))
    print(vars(args))

    # load dataset
    train_loader = get_dataloader(args)

    max_step = args.max_iters

    # load network and its pre-trained model
    model = get_model(args)

    # set optimizer
    optimizer = get_optimizer(args, model, max_step)

    # train
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    if args.network_type == 'cls':
        train_cls(train_loader, model, optimizer, max_step, args)
    elif args.network_type == 'eps':
        train_eps(train_loader, model, optimizer, max_step, args)
    else:
        raise Exception('No appropriate model type')