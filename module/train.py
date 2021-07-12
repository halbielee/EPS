import os

import torch
from torch.nn import functional as F

from eps import get_eps_loss
from util import pyutils


def train_cls(train_loader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_loader)
    for iteration in range(args.max_iters):
        for _ in range(args.iter_size):
            try:
                img_id, img, label = next(loader_iter)
            except:
                loader_iter = iter(train_loader)
                img_id, img, label = next(loader_iter)
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            pred = model(img)

            # Classification loss
            loss = F.multilabel_soft_margin_loss(pred, label)
            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'imps:%.1f' % ((iteration+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

            # validate(model, val_data_loader, epoch=ep + 1)
            timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_cls.pth'))


def train_eps(train_dataloader, model, optimizer, max_step, args):
    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_sal')
    timer = pyutils.Timer("Session started: ")
    loader_iter = iter(train_dataloader)
    for iteration in range(args.max_iters):
        for _ in range(args.iter_size):
            try:
                img_id, img, saliency, label = next(loader_iter)
            except:
                loader_iter = iter(train_dataloader)
                img_id, img, saliency, label = next(loader_iter)
            img = img.cuda(non_blocking=True)
            saliency = saliency.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            pred, cam = model(img)

            # Classification loss
            loss_cls = F.multilabel_soft_margin_loss(pred[:, :-1], label)

            loss_sal, fg_map, bg_map, sal_pred = get_eps_loss(cam, saliency, label, args.tau, args.alpha, intermediate=True)
            loss = loss_cls + loss_sal

            avg_meter.add({'loss': loss.item(),
                           'loss_cls': loss_cls.item(),
                           'loss_sal': loss_sal.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step-1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (iteration, args.max_iters),
                      'Loss_Cls:%.4f' % (avg_meter.pop('loss_cls')),
                      'Loss_Sal:%.4f' % (avg_meter.pop('loss_sal')),
                      'imps:%.1f' % ((iteration+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

            # validate(model, val_data_loader, epoch=ep + 1)
            timer.reset_stage()
    torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_cls.pth'))