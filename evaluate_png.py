import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import json


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        # mask = (label_true >= 0) & (label_true < self.num_classes)
        mask = (label_true >= 0) & (label_true < self.num_classes) & (label_pred < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        recall = np.diag(self.hist) / self.hist.sum(axis=1)
        # recall = np.nanmean(recall)
        precision = np.diag(self.hist) / self.hist.sum(axis=0)
        # precision = np.nanmean(precision)
        TP = np.diag(self.hist)
        TN = self.hist.sum(axis=1) - np.diag(self.hist)
        FP = self.hist.sum(axis=0) - np.diag(self.hist)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.num_classes), iu))

        return acc, recall, precision, TP, TN, FP, cls_iu, mean_iu, fwavacc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, default='/srv/PascalVOC/VOCdevkit/VOC2012/SegmentationClassAug/')
    parser.add_argument('--pred_dir', type=str)
    parser.add_argument('--datalist', type=str, default='data/train.txt')
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    mIOU = IOUMetric(num_classes=21)

    gt_dir = args.gt_dir
    list_dir = 'ImageSets/Segmentation/'
    ids = [i.split()[0].split('/')[2].split('.')[0].strip() for i in open(args.datalist) if not i.strip() == '']

    classes = np.array(('background',  # always index 0
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0), (0, 0, 0.5), (0.5, 0, 0.5), (0, 0.5, 0.5),
                (0.5, 0.5, 0.5), (0.25, 0, 0), (0.75, 0, 0), (0.25, 0.5, 0), (0.75, 0.5, 0), (0.25, 0, 0.5),
                (0.75, 0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5), (0, 0.25, 0), (0.5, 0.25, 0), (0, 0.75, 0),
                (0.5, 0.75, 0), (0, 0.25, 0.5)]
    values = [i for i in range(21)]
    color2val = dict(zip(colormap, values))

    import time

    st = time.time()
    for ind, img_id in tqdm(enumerate(ids)):
        img_path = os.path.join(gt_dir, img_id + '.png')
        pred_img_path = os.path.join(args.pred_dir, img_id + '.png')

        gt = Image.open(img_path)
        w, h = gt.size[0], gt.size[1]
        gt = np.array(gt, dtype=np.int32)  # shape = [h, w], 0-20 is classes, 255 is ingore boundary

        pred = Image.open(pred_img_path)
        pred = pred.crop((0, 0, w, h))
        pred = np.array(pred, dtype=np.int32)  # shape = [h, w]

        mIOU.add_batch(pred, gt)
        # print(img_id, ind)

    acc, recall, precision, TP, TN, FP, cls_iu, miou, fwavacc = mIOU.evaluate()

    mean_prec = np.nanmean(precision)
    mean_recall = np.nanmean(recall)

    result = {"Recall": ["{:.2f}".format(i) for i in recall.tolist()],
              "Precision": ["{:.2f}".format(i) for i in precision.tolist()],
              "Mean_Recall": mean_recall,
              "Mean_Precision": mean_prec,
              "IoU": cls_iu,
              "Mean IoU": miou,
              "TP": TP.tolist(),
              "TN": TN.tolist(),
              "FP": FP.tolist()}
    # result = {"Pixel Accuracy": acc,
    #           "Recall": recall.tolist(),
    #           "Precision": precision.tolist(),
    #           "TP": TP.tolist(),
    #           "TN": TN.tolist(),
    #           "FP": FP.tolist(),
    #           "Frequency Weighted IoU": fwavacc,
    #           "Mean IoU": miou,
    #           "Class IoU": cls_iu}

    with open(args.save_path, "w") as f:
        json.dump(result, f, indent=4, sort_keys=True)
    # print(acc, acc_cls, cls_iu, miou, fwavacc)
    print('mIOU = %s, time = %s s' % (miou, str(time.time() - st)))
