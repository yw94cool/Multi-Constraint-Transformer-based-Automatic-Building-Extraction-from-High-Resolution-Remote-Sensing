import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms

def trainer(args, model, snapshot_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from datasets.dataset import Building_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = Building_dataset(args=args, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=args.img_size)]))
    db_val = Building_dataset(args=args, split="val", transform=None)
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            '''Create sub_label'''
            assert len(label_batch.shape) == 3, "Label should be 2D."
            img_rows, img_cols = label_batch.shape[1:]

            x1_batch = np.zeros((label_batch.shape[0], img_cols // 8, img_rows // 8))
            for i in range(x1_batch.shape[0]):
                x1_batch[i,:,:] = cv2.resize(label_batch.numpy()[i,:,:], (img_cols // 8, img_rows // 8), interpolation=cv2.INTER_NEAREST)
            
            x2_batch = np.zeros((label_batch.shape[0], img_cols // 4, img_rows // 4))
            for i in range(x2_batch.shape[0]):
                x2_batch[i,:,:] = cv2.resize(label_batch.numpy()[i,:,:], (img_cols // 4, img_rows // 4), interpolation=cv2.INTER_NEAREST)
            
            x3_batch = np.zeros((label_batch.shape[0], img_cols // 2, img_rows // 2))
            for i in range(x3_batch.shape[0]):
                x3_batch[i,:,:] = cv2.resize(label_batch.numpy()[i,:,:], (img_cols // 2, img_rows // 2), interpolation=cv2.INTER_NEAREST)

            x1_batch = torch.from_numpy(x1_batch).float().to(device)
            x2_batch = torch.from_numpy(x2_batch).float().to(device)
            x3_batch = torch.from_numpy(x3_batch).float().to(device)
            
            image_batch, label_batch= image_batch.to(device), label_batch.to(device)
            x1, x2, x3, outputs = model(image_batch)

            '''Multi-Constrained Loss'''
            lossCE_main = ce_loss(outputs, label_batch[:].long())
            loss_sub1 = ce_loss(x1, x1_batch[:].long())
            loss_sub2 = ce_loss(x2, x2_batch[:].long())
            loss_sub3 = ce_loss(x3, x3_batch[:].long())
            loss = 0.4 * lossCE_main + 0.2 * loss_sub1 + 0.2 * loss_sub2 + 0.2 * loss_sub3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
        
        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"