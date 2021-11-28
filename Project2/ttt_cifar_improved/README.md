# TTT++

This is an official implementation for **TTT++: Improved Test-time Training**

TL;DR: Online Feature Alignment + Strong Self-supervised Learner &#129138; Robust Test-time Adaptation
> * Theoretically: new insights on the limitations and remedies of test-time training
> * Empirically: new state-of-the-art results on various robustness benchmarks

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

To download datasets:

```bash
sudo mkdir -p /data/cifar && cd /data/cifar
wget -O CIFAR-10-C.tar https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
tar -xvf CIFAR-10-C.tar
```

## Pre-trained Models

The checkpoint of the pre-train Resnet-50 can be downloaded (214MB) using the following command: 

```bash
mkdir -p results/cifar10_joint_resnet50 && cd results/cifar10_joint_resnet50
pip install gdown && gdown https://drive.google.com/uc?id=1TWiFJY_q5uKvNr9x3Z4CiK2w9Giqk9Dx && cd ../..
```

## Test-time Adaptation Scripts

Our proposed TTT++:

```bash
bash scripts/run_ttt++_cifar.sh
```

Prior state-of-the-art method TENT:

```bash
bash scripts/run_tent_cifar.sh
```

Prior state-of-the-art method SHOT:

```bash
bash scripts/run_shot_cifar.sh
```

## Feature Visualization

To generate the t-SNE figures for feature visualization, add `--tsne` to the above bash scripts.

## Results

The scripts above yield the following test results on the Cifar10-C under the snow corruption:

| Method | Error (%) |
| ------ | --------- |
|  Test  |   21.93   |
|  TENT  |   11.93   |
|  SHOT  |   13.39   |
| TTT++  |   **8.96**   |
