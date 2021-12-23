# CS-433-ML-SSH

### Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

### Pretrain

#### Pretrain resnet50

```python
python pretrain_resnet.py
```

#### Pretrain joint model

```python
python pretrain_joint.py
```

### Test-time training

#### TTT++

```bash
bash scripts/run_ttt++_cifar10.sh
```

#### TENT
```bash
bash scripts/run_tent_cifar10.sh
```

#### SHOT

```bash
bash scripts/run_shot_cifar10.sh
```

### Training Visualization

```bash
tensorboard -logdir save/iwildcam_tensorboard
```
