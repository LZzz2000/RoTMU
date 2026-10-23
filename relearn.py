import copy

from torch.utils.data import ConcatDataset, Subset

import config
import torch
from dataloader import get_dataloader, get_dataset, Customer_cifar10

from util import get_model, eval, Custom_dataset, train, eval_all_classes, eval_adverarial_attack

import os
import numpy as np
import random

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():

    opt = config.get_arguments().parse_args()
    set_seed(0)

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


    forget_indices = np.load(f'./exp/{opt.target_label}/forget_indices.npy')

    dataset_t_f = Subset(dataset_train, forget_indices)


    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)
    netC.load_state_dict(torch.load(f'./exp/{opt.target_label}/RoTMU_model.pth'))

    optimizerC = torch.optim.Adam(netC.parameters(), lr=1e-4)

    kk_list = []
    kk = eval(netC, dataset_t_f, False)
    kk_list.append(kk)

    subset_indices = np.random.permutation(len(dataset_t_f))[:20]
    ds = Subset(dataset_t_f, subset_indices)

    td = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=0,
                                       shuffle=False)

    original_model, _, _ = get_model(opt)
    original_model.load_state_dict(torch.load('./exp/model.pth'))
    target_accuracy = eval(original_model, dataset_t_f, False)

    epoch = 0
    while True:
        print("Epoch {}:".format(epoch + 1))
        train(netC , optimizerC, schedulerC, td)

        kk = eval(netC, dataset_t_f, False)
        kk_list.append(kk)
        if kk >= target_accuracy:
            break
        epoch += 1
        if epoch >= 100:
            break

    print(kk_list)


if __name__ == "__main__":
    main()
