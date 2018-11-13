import os
import argparse
import re
from collections import defaultdict, namedtuple


parser = argparse.ArgumentParser()
parser.add_argument('target_dir', type=str)
parser.add_argument('--plot', default=False, action='store_true')
args = parser.parse_args()

models = os.listdir(args.target_dir)
model_logs = defaultdict(list)
Log = namedtuple('Log', ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

for model in models:
    model_dir = os.path.join(args.target_dir, model)
    log_file = os.path.join(model_dir, 'output.log')

    with open(log_file, 'r') as fo:
        log = fo.read()

    epoch_pattern = r"Epoch (?P<epoch>\d+).*\n" \
                    r".*train Loss: (?P<train_loss>[0-1].\d+) Acc: (?P<train_acc>0.\d+).*\n" \
                    r".*val Loss: (?P<val_loss>[0-1].\d+) Acc: (?P<val_acc>0.\d+).*"

    for r in re.findall(epoch_pattern, log):
        model_logs[model].append(Log(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])))

    test_pattern = r"Best val Acc: (?P<best_val_acc>[0-1].\d+).*\n" \
                   r".*test Loss: (?P<test_loss>[0-1].\d+) Acc: (?P<test_acc>0.\d+)"
    test_logs = re.findall(test_pattern, log)
    if test_logs:
        print('{:<15}: best val acc: {:.2f}, test loss: {:.2f}, test acc: {:.2f}'.
              format(model, float(test_logs[0][0]), float(test_logs[0][1]), float(test_logs[0][2])))
    else:
        print('Cannot find test patter in {}'.format(model_dir))


if args.plot:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes[0][0].set_title('Train loss')
    axes[0][1].set_title('Train acc')
    axes[1][0].set_title('Val loss')
    axes[1][1].set_title('Val acc')
    for model, logs in model_logs.items():
        axes[0][0].plot([log.epoch for log in logs], [log.train_loss for log in logs])
        axes[0][1].plot([log.epoch for log in logs], [log.train_acc for log in logs])
        axes[1][0].plot([log.epoch for log in logs], [log.val_loss for log in logs])
        axes[1][1].plot([log.epoch for log in logs], [log.val_acc for log in logs])

    axes[0][0].legend(model_logs.keys())
    axes[0][1].legend(model_logs.keys())
    axes[1][0].legend(model_logs.keys())
    axes[1][1].legend(model_logs.keys())

    plt.show()
