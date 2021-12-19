import numpy as np
import torch
import torch.nn as nn
from utils.misc import *
from utils.rotation import rotate_batch


def load_resnet50(net, head, ssh, classifier, args):

    if args.ckpt:
        filename = args.resume + '/ckpt_epoch_{:d}.pth'.format(args.ckpt)
    else:
        filename = args.resume + '/ckpt.pth'
    ckpt = torch.load(filename)
    state_dict = ckpt['model']

    model_dict = net.state_dict()
    net_dict = {}
    head_dict = {}
    print("model_dict:")
    # print(model_dict)
    for k, v in model_dict.items():
        if k in ["head.fc.weight", "head.fc.bias", "fc.weight", "fc.bias"]:
            print(k)
        if k[:4] == "head":
            # k = k.replace("head.", "")
            # head_dict[k] = v
            pass
        else:
            k = k.replace("encoder.module.", "ext.")
            # k = k.replace("downsample", "shortcut")
            # k = k.replace("head.fc.", "fc.")
            net_dict[k] = v

    print("pretrained_dict:")
    # for k, v in state_dict.items():
    #     if "weight" in k:
    #         print(k)
    # for k, v in model_dict.items():
    #     if "weight" in k:
    #         print(k)
    pretrained_dict = {k:v for k, v in state_dict.items() if k in net_dict and "fc" not in k}
    net_dict["head.fc.weight"] = model_dict["head.fc.weight"]
    net_dict["head.fc.bias"] = model_dict["head.fc.bias"]

    # print(ckpt)
    # print(ckpt['model'])
    net_dict.update(pretrained_dict)
    print("net_dict:")
    for k, v in net_dict.items():
        if k in ["head.fc.weight", "head.fc.bias", "fc.weight", "fc.bias"]:
            print(k)
    net.load_state_dict(net_dict)

    # net_dict = {}
    for k, v in state_dict.items():
        if k[:4] == "head":
            k = k.replace("head.", "")
            head_dict[k] = v
    #     else:
    #         k = k.replace("encoder.module.", "ext.")
    #         k = k.replace("downsample", "shortcut")
    #         k = k.replace("fc.", "head.fc.")
    #         net_dict[k] = v
    head_dict["head.fc.weight"] = model_dict["head.fc.weight"]
    head_dict["head.fc.bias"] = model_dict["head.fc.bias"]
    head_dict["0.weight"] = model_dict["head.fc.weight"]
    head_dict["0.bias"] = model_dict["head.fc.bias"]
    head_dict["2.weight"] = model_dict["head.fc.weight"]
    head["2.bias"] = model_dict["head.fc.bias"]
    head_dict.load_state_dict(net_dict)
    #TODO make it a switch, will need to load to head in the future
    head.load_state_dict(head_dict)

    print('Loaded model trained jointly on Classification and SimCLR:', filename)


def load_ttt(net, head, ssh, classifier, args):
    filename = args.resume + '/{}_both.pth'.format(args.corruption)
    ckpt = torch.load(filename)
    net.load_state_dict(ckpt['net'])
    head.load_state_dict(ckpt['head'])
    print('Loaded updated model from', filename)


def corrupt_resnet50(ext, args):
    try:
        # SSL trained encoder
        simclr = torch.load(args.restore + '/simclr.pth')
        state_dict = simclr['model']

        ext_dict = {}
        for k, v in state_dict.items():
            if k[:7] == "encoder":
                k = k.replace("encoder.", "")
                ext_dict[k] = v
        ext.load_state_dict(ext_dict)

        print('Corrupted encoder trained by SimCLR')

    except:
        # Jointly trained encoder
        filename = args.resume + '/ckpt_epoch_{}.pth'.format(args.restore)

        ckpt = torch.load(filename)
        state_dict = ckpt['model']

        ext_dict = {}
        for k, v in state_dict.items():
            if k[:7] == "encoder":
                k = k.replace("encoder.", "")
                ext_dict[k] = v
        # import pdb; pdb.set_trace()
        # print_params(ext)
        ext.load_state_dict(ext_dict)
        print('Corrupted encoder jontly trained on Classification and SimCLR')


def build_resnet50(args):
    from models.BigResNet import SupConResNet, LinearClassifier
    from models.SSHead import ExtractorHead

    print('Building ResNet50...')
    classes = 186

    classifier = LinearClassifier(num_classes=classes).cuda()
    ssh = SupConResNet().cuda()
    head = ssh.head
    ext = ssh.encoder
    net = ExtractorHead(ext, classifier).cuda()
    return net, ext, head, ssh, classifier


def build_model(args):
    from models.ResNet import ResNetCifar as ResNet
    from models.SSHead import ExtractorHead
    print('Building model...')
    if args.dataset == 'cifar10':
        classes = 10
    elif args.dataset == 'cifar7':
        if not hasattr(args, 'modified') or args.modified:
            classes = 7
        else:
            classes = 10
    elif args.dataset == "cifar100":
        classes = 100

    if args.group_norm == 0:
        norm_layer = nn.BatchNorm2d
    else:
        def gn_helper(planes):
            return nn.GroupNorm(args.group_norm, planes)
        norm_layer = gn_helper

    if hasattr(args, 'detach') and args.detach:
        detach = args.shared
    else:
        detach = None
    net = ResNet(args.depth, args.width, channels=3, classes=classes, norm_layer=norm_layer, detach=detach).cuda()
    if args.shared == 'none':
        args.shared = None

    if args.shared == 'layer3' or args.shared is None:
        from models.SSHead import extractor_from_layer3
        ext = extractor_from_layer3(net)
        if not hasattr(args, 'ssl') or args.ssl == 'rotation':
            head = nn.Linear(64 * args.width, 4)
        elif args.ssl == 'contrastive':
            head = nn.Sequential(
                nn.Linear(64 * args.width, 64 * args.width),
                nn.ReLU(inplace=True),
                nn.Linear(64 * args.width, 16 * args.width)
            )
        else:
            raise NotImplementedError
    elif args.shared == 'layer2':
        from models.SSHead import extractor_from_layer2, head_on_layer2
        ext = extractor_from_layer2(net)
        head = head_on_layer2(net, args.width, 4)
    ssh = ExtractorHead(ext, head).cuda()

    if hasattr(args, 'parallel') and args.parallel:
        net = torch.nn.DataParallel(net)
        ssh = torch.nn.DataParallel(ssh)
    return net, ext, head, ssh

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def test(dataloader, model, sslabel=None):
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    model.eval()
    correct = []
    losses = []
    for batch_idx, (inputs, labels, meta) in enumerate(dataloader):
        if sslabel is not None:
            inputs, labels = rotate_batch(inputs, sslabel)
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.cpu())
            _, predicted = outputs.max(1)
            # acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            #
            # print("outputs and predicted")
            # print(predicted)
            # print(labels)
            correct.append(predicted.eq(labels).cpu())
    correct = torch.cat(correct).numpy()
    losses = torch.cat(losses).numpy()
    model.train()
    print("1-correct.mean(), correct, losses")
    print(1-correct.mean())
    print(correct)
    print(losses)
    return 1-correct.mean(), correct, losses


def test_grad_corr(dataloader, net, ssh, ext):
    criterion = nn.CrossEntropyLoss().cuda()
    net.eval()
    ssh.eval()
    corr = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        net.zero_grad()
        ssh.zero_grad()
        inputs_cls, labels_cls = inputs.cuda(), labels.cuda()
        outputs_cls = net(inputs_cls)
        loss_cls = criterion(outputs_cls, labels_cls)
        grad_cls = torch.autograd.grad(loss_cls, ext.parameters())
        grad_cls = flat_grad(grad_cls)

        ext.zero_grad()
        inputs, labels = rotate_batch(inputs, 'expand')
        inputs_ssh, labels_ssh = inputs.cuda(), labels.cuda()
        outputs_ssh = ssh(inputs_ssh)
        loss_ssh = criterion(outputs_ssh, labels_ssh)
        grad_ssh = torch.autograd.grad(loss_ssh, ext.parameters())
        grad_ssh = flat_grad(grad_ssh)

        corr.append(torch.dot(grad_cls, grad_ssh).item())
    net.train()
    ssh.train()
    return corr


def pair_buckets(o1, o2):
    crr = np.logical_and( o1, o2 )
    crw = np.logical_and( o1, np.logical_not(o2) )
    cwr = np.logical_and( np.logical_not(o1), o2 )
    cww = np.logical_and( np.logical_not(o1), np.logical_not(o2) )
    return crr, crw, cwr, cww


def count_each(tuple):
    return [item.sum() for item in tuple]


def plot_epochs(all_err_cls, all_err_ssh, fname, use_agg=True):
    import matplotlib.pyplot as plt
    if use_agg:
        plt.switch_backend('agg')

    plt.plot(np.asarray(all_err_cls)*100, color='r', label='classifier')
    plt.plot(np.asarray(all_err_ssh)*100, color='b', label='self-supervised')
    plt.xlabel('epoch')
    plt.ylabel('test error (%)')
    plt.legend()
    plt.savefig(fname)
    plt.close()
