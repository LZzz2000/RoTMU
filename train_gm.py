import copy

from torch.utils.data import ConcatDataset, Subset

import config
import torch
from dataloader import get_dataloader, get_dataset, Customer_cifar10

from util import get_model, eval, Custom_dataset, train, eval_all_classes, eval_adverarial_attack

import os
import numpy as np
import random

import torchvision.datasets as datasets

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():

    opt = config.get_arguments().parse_args()

    opt.num_classes = 10

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "cifar100":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    elif opt.dataset == "vgg":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")


    dataset_train = Customer_cifar10()
    forget_indices = dataset_train.forget_indices(opt.target_label)


    ratio = 0.98
    set_seed(0)
    shuffled_indices = np.random.permutation(forget_indices[0])
    split_point = int(len(shuffled_indices) * ratio)

    part1_indices = shuffled_indices[:split_point]
    part2_indices = shuffled_indices[split_point:]

    np.save(f'./exp/{opt.target_label}/forget_indices.npy', np.where(dataset_train.bool_array == True)[0][part2_indices])
    # np.save(f'./exp/{opt.target_label}/forget_indices.npy', part2_indices)
    #exit()
    dataset_t_nf = Subset(dataset_train, part1_indices)
    dataset_t_f = Subset(dataset_train, part2_indices)

    all_indices = set(range(len(dataset_train)))
    remaining_indices = list(all_indices - set(forget_indices[0]))
    dataset_t_r = Subset(dataset_train, remaining_indices)

    dataset_con = ConcatDataset((dataset_t_nf, dataset_t_r))

    # prepare model
    netC, _, _ = get_model(opt)
    optimizerC = torch.optim.SGD(
        netC.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
    )
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(
        optimizerC,
        milestones=[100, 150],
        gamma=0.1,
    )
    dl_train_con = torch.utils.data.DataLoader(dataset_con, batch_size=64, num_workers=opt.num_workers, shuffle=True)

    test_dataloader = get_dataloader(opt, False)

    for epoch in range(200):
        print("Epoch {}:".format(epoch + 1))
        train(netC , optimizerC, schedulerC, dl_train_con)

        eval(netC, dl_train_con , opt)
        eval(netC, dataset_t_f, opt)

    opt.save_dir = './exp/'
    torch.save(netC.state_dict(), os.path.join(opt.save_dir, f'{opt.target_label}/Gmodel.pth'))


if __name__ == "__main__":
    main()
