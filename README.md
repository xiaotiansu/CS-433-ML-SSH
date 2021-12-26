# ml-project-2-ssh2

Supervised by VITA: https://www.epfl.ch/labs/vita/

This code repo was modified based on VITA's code repo of training ttt++ algorithm on cifar10/100 dataset.

Environment requirements: pytorch >= 1.8.1, CUDA >= 11.1

### Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

To setup data folder to store dataset:
```bash

export DATADIR= {your path}  # if it is empty, dataset will be automatically downloaded from wilds. It takes about half an hour.
```

### Pretrain

#### Pretrain resnet50

```python
python ttt_wilds/pretrain_resnet.py
```

#### Pretrain joint model

```python
python ttt_wilds/pretrain_joint.py
```

### Test-time training

#### TTT++

```bash
bash ttt_wilds/scripts/run_ttt++_wilds.sh
```

#### TENT
```bash
bash ttt_wilds/scripts/run_tent_wilds.sh
```

#### SHOT

```bash
bash ttt_wilds/scripts/run_shot_wilds.sh
```

### Visualization training result on tensorboard

```bash
tensorboard -logdir ttt_wilds/save/iwildcam_tensorboard
```
