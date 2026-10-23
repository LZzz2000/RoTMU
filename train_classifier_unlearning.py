from torch.utils.data import ConcatDataset, Subset, Dataset
import torch.utils.data as Data
import config
import torch
from classifier_models import pure_model
from dataloader import get_dataloader, get_dataset, Customer_cifar10
from un import noise_train_SAM
from util import get_model, eval, Custom_dataset, train, trainb, evalb, eval_adverarial_attack, \
    eval_rl, Custom_dataset_for_unlearning
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import random
from torch.optim import Optimizer

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

def outline(data):
    Q1 = torch.quantile(data, 0.25)
    Q3 = torch.quantile(data, 0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_low = data < lower_bound
    outliers_high = data > upper_bound

    data = torch.where(data < lower_bound, lower_bound, data)
    data = torch.where(data > upper_bound, upper_bound, data)

    return data


i = 0

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    opt = config.get_arguments().parse_args()

    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "imagenet":
        opt.num_classes = 100
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    elif opt.dataset == "vgg":
        opt.num_classes = 10
    elif opt.dataset == "cifar100":
        opt.num_classes = 100
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "imagenet":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    elif opt.dataset == "vgg":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    elif opt.dataset == "cifar100":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    dataset_test = get_dataset(opt, train=False)
    dataset_test = Custom_dataset(dataset_test)
    dataset_test.filter([i for i in range(opt.num_classes) if i != opt.target_label])


    dataset_train = Customer_cifar10()
    forget_indices = dataset_train.forget_indices(opt.target_label)


    ratio = 0.98

    set_seed(0)
    shuffled_indices = np.random.permutation(forget_indices[0])
    split_point = int(len(shuffled_indices) * ratio)


    part1_indices = shuffled_indices[:split_point]
    part2_indices = shuffled_indices[split_point:]

    np.save(f'./exp/{opt.target_label}/forget_indices.npy', np.where(dataset_train.bool_array == True)[0][part2_indices])
    np.save(f'./exp/{opt.target_label}/retrain_indices.npy', np.where(dataset_train.bool_array == True)[0][part1_indices])
    # np.save(f'./exp/{opt.target_label}/forget_indices.npy', part2_indices)
    # np.save(f'./exp/{opt.target_label}/retrain_indices.npy', part1_indices)
    #exit()
    dataset_t_nf = Subset(dataset_train, part1_indices)
    dataset_t_f = Subset(dataset_train, part2_indices)

    all_indices = set(range(len(dataset_train)))
    remaining_indices = list(all_indices - set(forget_indices[0]))
    dataset_t_r = Subset(dataset_train, remaining_indices)

    from un import get_feature
    # prepare model

    # original model是finetune model的gold model，gold model是original model的gold model
    # original model
    netG, _, _ = get_model(opt)
    netG.load_state_dict(torch.load(f'./exp/model.pth'))

    # finetune model
    netC, optimizerC, schedulerC = get_model(opt)
    netC.load_state_dict(torch.load(f'./exp/{opt.target_label}/test_finetune.pth'))

    # gold model
    netGG, _, _ = get_model(opt)
    # real gold model
    netGG.load_state_dict(torch.load(f'./exp/{opt.target_label}/Gmodel.pth'))

    model = pure_model().to('cuda')

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
    )

    netO, _, _ = get_model(opt)
    netO.load_state_dict(torch.load(f'./exp/{opt.target_label}/Curriculum_model.pth'))

    netTO, _, _ = get_model(opt)
    netTO.load_state_dict(torch.load(f'./exp/{opt.target_label}/Curriculum_model.pth'))

    netUL, _, _ = get_model(opt)

    print("here are evaluations")

    dataset_T, dataset_F = eval(netG, dataset_test)

    preds_R = get_feature(netC, dataset_t_nf, logits=False)

    preds_R = torch.nn.functional.normalize(preds_R, p=2, dim=1)


    ds_T = Custom_dataset(dataset_T)
    preds_T = get_feature(netC, ds_T, logits=False)
    label_T = torch.ones(len(preds_T))

    td_t = torch.utils.data.DataLoader(ds_T, batch_size=64, num_workers=0,
                                       shuffle=False)
    loss_t = eval_rl(netTO, td_t)

    ds_F = Custom_dataset(dataset_F)
    preds_F = get_feature(netC, ds_F, logits=False)
    label_F = torch.zeros(len(preds_F))
    td_f = torch.utils.data.DataLoader(ds_F, batch_size=64, num_workers=0,
                                       shuffle=False)
    loss_f = eval_rl(netTO, td_f)

    preds_con = torch.cat((preds_T, preds_F), dim=0)
    label_con = torch.cat((label_T, label_F), dim=0)

    print(torch.sum(label_con))

    d_F = []
    L_F = []

    for i in range(len(preds_F)):
        preds_t = preds_F[i].unsqueeze(0)
        # print(preds_t.shape)
        preds_t = preds_t.repeat(len(preds_R), 1)
        preds_t = torch.nn.functional.normalize(preds_t, p=2, dim=1)
        dis = torch.pairwise_distance(preds_t, preds_R, p=2)
        ind = torch.argsort(dis)
        dis = torch.mean(dis[ind[:5]])
        d_F.append(dis)
        L_F.append(torch.tensor(0))

    d_F = torch.stack(d_F)
    L_F = torch.stack(L_F)
    print(torch.mean(d_F))

    d_T = []
    L_T = []

    for i in range(len(preds_T)):
        preds_t = preds_T[i].unsqueeze(0)
        # print(preds_t.shape)
        preds_t = preds_t.repeat(len(preds_R), 1)
        preds_t = torch.nn.functional.normalize(preds_t, p=2, dim=1)
        dis = torch.pairwise_distance(preds_t, preds_R, p=2)
        ind = torch.argsort(dis)
        dis = torch.mean(dis[ind[:5]])
        d_T.append(dis)
        L_T.append(torch.tensor(1))

    d_T = torch.stack(d_T)
    L_T = torch.stack(L_T)
    print(torch.mean(d_T))

    adro_T = eval_adverarial_attack(netC, td_t, eps=4 / 255)
    print(torch.mean(adro_T))
    adro_F = eval_adverarial_attack(netC, td_f, eps=4 / 255)
    print(torch.mean(adro_F))
    # exit()

    d_1 = torch.cat((d_T, d_F), dim=0).cpu()
    d_2 = torch.cat((adro_T, adro_F), dim=0).cpu()
    d_3 = torch.cat((loss_t, loss_f), dim=0).cpu()

    ind = torch.argsort(d_1, descending=True)

    d_1 = outline(d_1)
    d_2 = outline(d_2)
    d_3 = outline(d_3)

    max, min = torch.max(d_1), torch.min(d_1)
    d_1 = (d_1 - min) / (max - min)
    max, min = torch.max(d_2), torch.min(d_2)
    d_2 = (d_2 - min) / (max - min)
    max, min = torch.max(d_3), torch.min(d_3)
    d_3 = (d_3 - min) / (max - min)

    d1_con = d_1
    d2_con = d_2
    d3_con = d_3

    # d1_con = torch.cat((d_1[:int(3.5 * lenth)], d_1[-lenth:]), dim=0)
    # d2_con = torch.cat((d_2[:int(3.5 * lenth)], d_2[-lenth:]), dim=0)
    # d3_con = torch.cat((d_3[:int(3.5 * lenth)], d_3[-lenth:]), dim=0)

    feature_con = torch.stack((d1_con.cpu(), d2_con.cpu(), d3_con.cpu()), dim=-1)

    # feature_con = torch.stack((d2_con.cpu(), d3_con.cpu()), dim=-1)
    # feature_con = d3_con.cpu().unsqueeze(-1)

    print(feature_con.shape)
    print(label_con.shape)

    # dataset_train = Data.TensorDataset(feature_con, label_con)
    # train_dl = torch.utils.data.DataLoader(dataset_train, batch_size=64, num_workers=0,
    #                                        shuffle=True)

    dataset_train = Data.TensorDataset(feature_con, label_con)

    label_count = torch.bincount(label_con.long(), minlength=2)
    class_weight = 1.0 / label_count.float()
    sample_weight = class_weight[label_con.long()].double()

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weight,
        num_samples=len(sample_weight),
        replacement=True,
    )

    train_dl = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,
        num_workers=0,
        sampler=sampler,
    )

    # dataset_T, dataset_F = eval(netGG, dataset_t_f)

    # acc_f_gm = 100.0 * len(dataset_T) / (len(dataset_T) + len(dataset_F))
    # print("ACC Forget set GM:{}".format(acc_f_gm))

    from util import getF
    label_test = getF(netGG, dataset_t_f)
    print(torch.sum(label_test))

    preds_test = get_feature(netG, dataset_t_f, logits=False)

    loss_test = eval_rl(netO, dataset_t_f)

    d_test = []
    for i in range(len(preds_test)):
        preds_t = preds_test[i].unsqueeze(0)
        # print(preds_t.shape)
        preds_t = preds_t.repeat(len(preds_R), 1)
        preds_t = torch.nn.functional.normalize(preds_t, p=2, dim=1)
        dis = torch.pairwise_distance(preds_t, preds_R, p=2)
        ind = torch.argsort(dis)
        dis = torch.mean(dis[ind[:5]])
        d_test.append(dis)
    d_test = torch.stack(d_test)

    adro_test = eval_adverarial_attack(netG, dataset_t_f, eps=4 / 255)

    d_test_1 = d_test.cpu()
    d_test_2 = adro_test.cpu()
    d_test_3 = loss_test.cpu()

    d_test_1 = outline(d_test_1)
    d_test_2 = outline(d_test_2)
    d_test_3 = outline(d_test_3)

    max, min = torch.max(d_test_1), torch.min(d_test_1)
    d_test_1 = (d_test_1 - min) / (max - min)
    max, min = torch.max(d_test_2), torch.min(d_test_2)
    d_test_2 = (d_test_2 - min) / (max - min)

    max, min = torch.max(d_test_3), torch.min(d_test_3)
    d_test_3 = (d_test_3 - min) / (max - min)

    d1_test_con = d_test_1
    d2_test_con = d_test_2

    feature_test_con = torch.stack((d1_test_con.cpu(), d2_test_con.cpu(), d_test_3.cpu()), dim=-1)
    label_test_con = label_test

    dataset_test = Data.TensorDataset(feature_test_con, label_test_con)
    test_dl = torch.utils.data.DataLoader(dataset_test, batch_size=64, num_workers=0,
                                          shuffle=False)


    if os.path.exists(f'./exp/{opt.target_label}/classifier.pth'):
        model.load_state_dict(torch.load(f'./exp/{opt.target_label}/classifier.pth'))
    else:
        for epoch in range(100):
            print("epoch is {}".format(epoch))
            trainb(model, optimizer, train_dl, opt)
            if epoch % 1 == 0:
                acc_clean, acc_t, acc_f = evalb(model, train_dl, False)
                acc_clean2, acc_t2, acc_f2 = evalb(model, test_dl, False)

        import csv
        with open('model.csv', 'a+') as file:
            writer = csv.writer(file)
            writer.writerow(
                [opt.target_label, acc_clean, float(acc_t), float(acc_f), acc_clean2, float(acc_t2),
                 float(acc_f2)])

        torch.save(model.state_dict(), f'./exp/{opt.target_label}/classifier.pth')

    torch.save(model.state_dict(), f'./exp/{opt.target_label}/classifier.pth')

    Flag = evalb(model, test_dl)

    Flag = np.array(Flag)

    dataset_per_f1 = Custom_dataset(dataset_t_f)
    ds_gT, ds_gF = dataset_per_f1.spilt(Flag)
    ds_gF = Custom_dataset(ds_gF)

    # ds_gF.randomlabel(opt.target_label)

    ds_gT = Custom_dataset(ds_gT)
    ds_gF = Custom_dataset(ds_gF)

    shuffled_indices = np.random.permutation(len(dataset_t_r))
    split_point = int(len(shuffled_indices) * 0.2)

    part1_indices = shuffled_indices[:split_point]
    dataset_t_r_percent = Subset(dataset_t_r, part1_indices)

    ds_gTu = Custom_dataset_for_unlearning(ds_gT, Flag=1)
    ds_gFu = Custom_dataset_for_unlearning(ds_gF, Flag=2)
    print(len(ds_gT))
    print(len(ds_gF))

    dataset_gcon = ConcatDataset(
        [ds_gTu, ds_gFu, Custom_dataset_for_unlearning(dataset_t_nf), Custom_dataset_for_unlearning(dataset_t_r)])
    print(len(dataset_gcon))
    dl_train_gcon = torch.utils.data.DataLoader(dataset_gcon, batch_size=64, num_workers=1, shuffle=True)
    td = get_dataloader(opt, False)

    loss_easy = torch.mean(eval_rl(netC, ds_T))
    loss_hard = torch.mean(eval_rl(netC, ds_F))
    print(loss_easy)
    print(loss_hard)
    eval(netC, dataset_t_r)
    model.eval()
    eval(netC, ds_T, reloss=True)
    eval(netC, ds_F, reloss=True)
    eval(netC, ds_gT, reloss=True)

    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    print("before:")
    eval(netC, dataset_t_f, opt)
    eval(netC, train_dl, opt)
    eval(netC, test_dl, opt)

    for epoch in range(5):
        print("Epoch {}:".format(epoch + 1))
        optimizerC = SAM(netC.parameters(), base_optimizer=torch.optim.Adam, lr=1e-3, rho=0.01)
        noise_train_SAM(netC, optimizerC, dl_train_gcon)

        if epoch % 1 == 0:
            eval(netC, dataset_t_f, opt)
            eval(netC, train_dl, opt)
            eval(netC, test_dl, opt)


    path = f'./exp/{opt.target_label}/RoTMU_model.pth'
    lists = []
    for i in range(len(dataset_t_f)):
        lists.append(dataset_t_f[i][0])


    torch.save(netC.state_dict(), path)
    '''
    import csv
    with open('data.csv', 'a+') as file:
        writer = csv.writer(file)
        writer.writerow(
            [opt.target_label, float(acc_f_gm), float(acc_test_gm), float(best_acc_f), float(best_acc_r), float(ar),
             len(ds_gT), len(ds_gF)])

    '''


if __name__ == "__main__":
    main()
