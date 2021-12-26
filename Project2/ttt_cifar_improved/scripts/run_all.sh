#! /usr/bin/env bash

# bash scripts/run_ttt++_cifar.sh snow both 16
# bash scripts/run_ttt++_cifar.sh snow both 32
# bash scripts/run_ttt++_cifar.sh snow both 64
# bash scripts/run_ttt++_cifar.sh snow both 128
# bash scripts/run_ttt++_cifar.sh snow both 256
# bash scripts/run_ttt++_cifar.sh snow both 512

# bash scripts/run_ttt++_cifar.sh snow ssl 16
# bash scripts/run_ttt++_cifar.sh snow ssl 32
# bash scripts/run_ttt++_cifar.sh snow ssl 64
# bash scripts/run_ttt++_cifar.sh snow ssl 128
# bash scripts/run_ttt++_cifar.sh snow ssl 256
# bash scripts/run_ttt++_cifar.sh snow ssl 512

# bash scripts/run_ttt++_cifar.sh snow align 16
# bash scripts/run_ttt++_cifar.sh snow align 32
# bash scripts/run_ttt++_cifar.sh snow align 64
# bash scripts/run_ttt++_cifar.sh snow align 128
# bash scripts/run_ttt++_cifar.sh snow align 256
# bash scripts/run_ttt++_cifar.sh snow align 512

# bash scripts/run_shot_cifar.sh snow shot 16
# bash scripts/run_shot_cifar.sh snow shot 32
# bash scripts/run_shot_cifar.sh snow shot 64
# bash scripts/run_shot_cifar.sh snow shot 128
# bash scripts/run_shot_cifar.sh snow shot 256
# bash scripts/run_shot_cifar.sh snow shot 512

# bash scripts/run_tent_cifar.sh snow tent 16
# bash scripts/run_tent_cifar.sh snow tent 32
# bash scripts/run_tent_cifar.sh snow tent 64
# bash scripts/run_tent_cifar.sh snow tent 128
# bash scripts/run_tent_cifar.sh snow tent 256
# bash scripts/run_tent_cifar.sh snow tent 512

# bs=1024
# factor=8
# qs=$((${bs}*${factor}))

# bash scripts/run_ttt++_cifar.sh snow ssl $bs $qs
# bash scripts/run_ttt++_cifar.sh snow align $bs $qs
# bash scripts/run_ttt++_cifar.sh snow both $bs $qs

python pretrain_resnet.py
python pretrain_joint.py

bash scripts/run_ttt++_cifar10.sh
bash scripts/run_tent_cifar10.sh
bash scripts/run_shot_cifar10.sh
