import os
import argparse
import re
from operator import itemgetter
from collections import defaultdict, namedtuple
import numpy as np


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


Log = namedtuple('Log', ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])


parser = argparse.ArgumentParser()
parser.add_argument('target_dir', type=str)
parser.add_argument('--plot', default=False, action='store_true')
parser.add_argument('--window_size', type=int, default=5)
args = parser.parse_args()

models = os.listdir(args.target_dir)
model_logs = defaultdict(list)
test_logs = list()
for model in models:
    model_dir = os.path.join(args.target_dir, model)
    log_file = os.path.join(model_dir, 'output.log')

    with open(log_file, 'r') as fo:
        log = fo.read()

    epoch_pattern = r"Epoch (?P<epoch>\d+).*\n" \
                    r".*train Loss: (?P<train_loss>\d+.\d+) Acc: (?P<train_acc>0.\d+).*\n" \
                    r".*val Loss: (?P<val_loss>\d+.\d+) Acc: (?P<val_acc>0.\d+).*"

    for r in re.findall(epoch_pattern, log):
        model_logs[model].append(Log(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])))

    test_pattern = r"Best val Acc: (?P<best_val_acc>[0-1].\d+).*\n" \
                   r".*test Loss: (?P<test_loss>\d+.\d+) Acc: (?P<test_acc>0.\d+)"
    test_log = re.findall(test_pattern, log)
    if test_log:
        test_logs.append((model, float(test_log[0][0]), float(test_log[0][1]), float(test_log[0][2])))
    else:
        print('Cannot find test patter in {}'.format(model_dir))

test_logs = sorted(test_logs, key=itemgetter(3), reverse=True)
for test_log in test_logs:
    print('{:<15}: best val acc: {:.4f}, test loss: {:.4f}, test acc: {:.4f}'.format(*test_log))


if args.plot:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes[0][0].set_title('Train loss')
    axes[0][1].set_title('Train acc')
    axes[1][0].set_title('Val loss')
    axes[1][1].set_title('Val acc')
    for model, logs in model_logs.items():
        epochs = [log.epoch for log in logs]
        train_loss = [log.train_loss for log in logs]
        train_acc = [log.train_acc for log in logs]
        val_loss = [log.val_loss for log in logs]
        val_acc = [log.val_acc for log in logs]

        train_loss_smooth = running_mean(train_loss, args.window_size)
        train_acc_smooth = running_mean(train_acc, args.window_size)
        val_loss_smooth = running_mean(val_loss, args.window_size)
        val_acc_smooth = running_mean(val_acc, args.window_size)
        epochs_smooth = epochs[:len(train_acc_smooth)]

        axes[0][0].plot(epochs_smooth, train_loss_smooth)
        axes[0][1].plot(epochs_smooth, train_acc_smooth)
        axes[1][0].plot(epochs_smooth, val_loss_smooth)
        axes[1][1].plot(epochs_smooth, val_acc_smooth)

    axes[0][0].legend(model_logs.keys())
    axes[0][1].legend(model_logs.keys())
    axes[1][0].legend(model_logs.keys())
    axes[1][1].legend(model_logs.keys())

    plt.show()
