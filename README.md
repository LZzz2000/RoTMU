# RoTMU: Robust Twin Machine Unlearning

This is the offical implementation of paper: **"Aligned and Robust Data Forgetting through Twin Machine Unlearning"**



## Getting Started

### Installation

**Step 1: Set up the environment**

```bash
cd RoTMU
conda env create -f env.yaml
conda activate RoTMU
```

**Step 2: Download the dataset**

Download the CIFAR-10 dataset from: http://www.cs.toronto.edu/~kriz/cifar.html



### Run the Code

#### Train the original model

```bash
python train_original.py --dataset cifar10
```

#### Train the gold model

```bash
python train_gm.py --dataset cifar10 --target_label 1
```

#### Prepare for unlearning

- Train a model from scratch for 2 epochs for loss feature

```bash
python train_curriculum_model.py --dataset cifar10 --target_label 1
```

- Finetune original model on val dataset for the other two features

```bash
python train_test_finetune_model.py --dataset cifar10 --target_label 1
```

#### Train the generalization binary classifier and perform unlearning

```bash
python train_classifier_unlearning.py --dataset cifar10 --target_label 1
```

#### Relearn attack

```bash
python relearn.py --dataset cifar10 --target_label 1
```

