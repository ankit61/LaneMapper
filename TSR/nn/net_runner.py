import net
import torch.nn as nn
import torch

import sys
from pathlib import Path
import os

sys.path.append(str(Path(os.path.realpath(__file__)).parents[1]))

from utils import constants, average_meter
import time
import dataset
import tensorboardX
from torchvision import transforms

'''
inspired by: https://github.com/pytorch/examples/blob/master/imagenet/main.py
'''
class WeightedBCELoss():
    def __init__(self, weight = 3):
      self.__w = weight
      
    def __call__(self, pred, target):
      return (-self.__w * target * torch.log(pred) - (1 - target) * torch.log(1 - pred)).mean()

class TSRNetRunner:
    def __init__(self, 
                 data_root          = constants.GTSRB_ROOT, 
                 num_traffic_signs  = constants.NUM_TRAFFIC_SIGNS,
                 batch_size         = constants.BATCH_SIZE,
                 train_epochs       = constants.TRAIN_EPOCHS, 
                 lr                 = constants.LR,
                 momentum           = constants.MOMENTUM,
                 threshold_prob     = constants.THRESHOLD_PROB,
                 load_model_path    = '',
                 should_log_tensorboard = False):

        self.__net                  = net.TSRNet(num_traffic_signs).cuda() if torch.cuda.is_available() else net.TSRNet(num_traffic_signs)
        
        if(load_model_path):
            self.__net.load_state_dict(torch.load(load_model_path)['state_dict'])
        
        self.__loss_fun             = WeightedBCELoss()
        self.__print_freq           = constants.PRINT_FREQ
        self.__writer               = tensorboardX.SummaryWriter(constants.TENSORBOARD_LOG_DIR)
        self.__batch_size           = batch_size
        self.__num_traffic_signs    = num_traffic_signs
        self.__train_path           = os.path.join(data_root, 'train')
        self.__val_path             = os.path.join(data_root, 'val')
        self.__test_path            = os.path.join(data_root, 'test')
        self.__train_epochs         = train_epochs
        self.__lr                   = lr
        self.__momentum             = momentum
        self.__threshold_prob       = threshold_prob
        self.__should_log_tensorboard = should_log_tensorboard

        self.__train_transforms_single_class = constants.TRAIN_TRANSFORMS_SINGLE_CLASS
        
        self.__train_transforms_multi_class  = constants.TRAIN_TRANSFORMS_MULTI_CLASS

        self.__test_transforms = constants.TEST_TRANSFORMS

    def train(self, train_loader, optimizer, epoch):
        self.__net.train()

        loss_meter              = average_meter.AverageMeter()
        batch_load_time_meter   = average_meter.AverageMeter()
        batch_time_meter        = average_meter.AverageMeter()
        percent_ones_meter      = average_meter.AverageMeter()

        start_time = time.time()

        for i, (imgs, gt_class_probs) in enumerate(train_loader):
                batch_load_time_meter.update(time.time() - start_time)
                probs = self.__net(imgs)

                loss = self.__loss_fun(probs, gt_class_probs)
                percent_ones = float(probs.sum()) / probs.numel()
                
                loss_meter.update(loss.item())
                percent_ones_meter.update(percent_ones)
                if(self.__should_log_tensorboard):
                    self.__writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + i)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_time_meter.update(time.time() - start_time)

                if i % self.__print_freq == 0:
                    print(
                        'Epoch[{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader)),
                        'loss:', loss_meter.last(), loss_meter.avg(), '\t'
                        'Percent Ones:', percent_ones_meter.last(), percent_ones_meter.avg(), '\t'
                        'Time:', batch_time_meter.last(), batch_time_meter.avg(), '\t'
                        'Data:', batch_load_time_meter.last(), batch_load_time_meter.avg(), '\t'
                        )
                
                start_time = time.time()
                
               
        print('Epoch[{0}][{1}/{2}]\t'.format(epoch, len(train_loader), len(train_loader)),
              'loss:', loss_meter.last(), loss_meter.avg(), '\t'
              'Percent Ones:', percent_ones_meter.last(), percent_ones_meter.avg(), '\t'
              'Time:', batch_time_meter.last(), batch_time_meter.avg(), '\t'
              'Data:', batch_load_time_meter.last(), batch_load_time_meter.avg(), '\t'
             )

    def __save_checkpoint(self, state, file_name = 'checkpoint.pth'):
        torch.save(state, file_name)

    def test(self, test_loader, epoch):
        self.__net.eval()
        
        loss_meter         = average_meter.AverageMeter()
        acc_meter          = average_meter.AverageMeter()
        recall_meter       = average_meter.AverageMeter()
        percent_ones_meter = average_meter.AverageMeter()

        with torch.no_grad():
            for i, (imgs, gt_class_probs) in enumerate(test_loader):
                probs = self.__net(imgs)

                loss = self.__loss_fun(probs, gt_class_probs)
                acc, recall  = self.__get_accuracy_and_recall(probs, gt_class_probs, epoch * len(test_loader) + i)
                percent_ones = float(probs.sum()) / probs.numel()
                
                loss_meter.update(loss.item())
                acc_meter.update(acc)
                recall_meter.update(recall)
                percent_ones_meter.update(percent_ones)
                
                if(self.__should_log_tensorboard):
                    self.__writer.add_scalar('val/loss', loss.item(), epoch * len(test_loader) + i)
                    self.__writer.add_scalar('val/acc', acc, epoch * len(test_loader) + i)
                    self.__writer.add_scalar('val/recall', recall, epoch * len(test_loader) + i)

                if i % self.__print_freq == 0:
                    print(
                        'Epoch[{0}][{1}/{2}]\t'.format(
                            epoch, i, len(test_loader)
                        ),
                        'loss:', loss_meter.last(), loss_meter.avg(), '\t'
                        'Acc:', acc_meter.last(), acc_meter.avg(), '\t'
                        'Recall:', recall_meter.last(), recall_meter.avg(), '\t'
                        'Percent Ones:', percent_ones_meter.last(), percent_ones_meter.avg(), '\t'
                    )
            
            print('Epoch[{0}][{1}/{2}]\t'.format(
                    epoch, len(test_loader), len(test_loader)
                 ),
                  'loss:', loss_meter.last(), loss_meter.avg(), '\t'
                  'Acc:', acc_meter.last(), acc_meter.avg(), '\t'
                  'Recall:', recall_meter.last(), recall_meter.avg(), '\t'
                  'Percent Ones:', percent_ones_meter.last(), percent_ones_meter.avg(), '\t'
                 )
            
            return acc_meter.avg()

    def __get_accuracy_and_recall(self, pred, gt, global_step):
        pred_ts = pred >= self.__threshold_prob
        gt_ts   = gt >= self.__threshold_prob
        
        tp = (pred_ts * gt_ts).sum()
        fp = (pred_ts * ~gt_ts).sum()
        fn = (~pred_ts * gt_ts).sum()
        tn = (~pred_ts * ~gt_ts).sum()
        
        if(self.__should_log_tensorboard):
            self.__writer.add_scalar('val/tp', tp, global_step)
            self.__writer.add_scalar('val/fp', fp, global_step)
            self.__writer.add_scalar('val/fn', fn, global_step)
            self.__writer.add_scalar('val/tn', tn, global_step)
        
        return float(tp + tn) / float(tp + fp + tn + fn), float(tp) / float(tp + fn)

    def run(self):
        train_dataset = dataset.TSRDataset(self.__train_transforms_single_class, self.__train_transforms_multi_class, self.__num_traffic_signs, root_dir = self.__train_path)
        train_loader  = torch.utils.data.DataLoader(
            train_dataset, batch_size = self.__batch_size, shuffle=True
        )

        val_dataset = dataset.TSRDataset(self.__test_transforms, self.__test_transforms, self.__num_traffic_signs, root_dir = self.__val_path)
        val_loader  = torch.utils.data.DataLoader(
            val_dataset, batch_size = self.__batch_size, shuffle=True
        )

        test_dataset = dataset.TSRDataset(self.__test_transforms, self.__test_transforms, self.__num_traffic_signs, root_dir = self.__test_path)
        test_loader  = torch.utils.data.DataLoader(
            test_dataset, batch_size = self.__batch_size, shuffle=True
        )

        optimizer = torch.optim.SGD(self.__net.parameters(), self.__lr, momentum = self.__momentum)

        best_acc = 0
        for epoch in range(self.__train_epochs):
            self.train(train_loader, optimizer, epoch)
            acc = self.test(val_loader, epoch)
            if(acc > best_acc):
                self.__save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.__net.state_dict()
                }, 'checkpoint_' + str(epoch) + '.pth')
        
        self.test(test_loader, 0)
