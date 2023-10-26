import os
import sys
sys.path.append(os.path.abspath('/workspace/wjj910'))
from data_utils import get_loader
from data_utils_temp import get_loader_under_sample
import argparse
import torch
import monai
from trainer import *
import random
import numpy as np
import os
# from utils import *
# from resnet import resnet50
import shutil
from monai.networks.layers.factories import Act, Norm

parser = argparse.ArgumentParser(description="AUS resnet classification")
parser.add_argument("--epochs", default=300)
parser.add_argument("--log_dir", default="/workspace/wjj910/clf/results_liver/", type=str)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--warmup_epochs", default=5, type=int)
parser.add_argument("--scheduler_switch", default=True, type=bool)
parser.add_argument("--lr_scheduler", default="poly_lr", type=str)
parser.add_argument("--optimizer", default="Adamw")
parser.add_argument("--min_lr", default=1e-5, type=float)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--test_mode", default=False, type=bool)
parser.add_argument('--cuda_visible_devices', default='2', type=str, help='cuda_visible_devices')
parser.add_argument('--pretrain', default=False, type=bool, help='cuda_visible_devices')
parser.add_argument("--organ", default="liver", type=str, help='liver, spleen, kidney')
parser.add_argument("--val_fold", default=0, type=int)
parser.add_argument("--patch", default=0, type=int)
parser.add_argument("--temp", default=0, type=int)
parser.add_argument("--under_sample", default=0, type=int)
# parser.add_argument("--weight", default=0, type=int)
# parser.add_argument("--temp", default=0, type=int)



def main():
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_visible_devices
    
    main_worker(args=args)
    
def main_worker(args):
    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)
    # print_args(args)

    device = torch.device('cuda')
    if args.under_sample == 0:
        train_loader, val_loader, train_len, val_len = get_loader(args)
    else:
        train_loader, val_loader, train_len, val_len = get_loader_under_sample(args)

    os.makedirs(args.log_dir, mode=0o777, exist_ok=True) 

    # model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=3).to(device)
    # model = monai.networks.nets.resnet101(spatial_dims=3, n_input_channels=1, num_classes=2).to(device)
    # model = monai.networks.nets.resnet50(spatial_dims=2, n_input_channels=1, num_classes=2, pretrained=args.pretrain).to(device)
    # model = monai.networks.nets.resnet50(spatial_dims=2, n_input_channels=1, num_classes=5, pretrained=args.pretrain).to(device)
    # model = resnet50(spatial_dims=3, n_input_channels=1, num_classes=3).to(device)
    print("check")
    print("renset_50")
    # model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, num_classes=3, conv1_t_stride=(2,2,1), norm_type=Norm.GROUP).to(device)
    model = monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=1, num_classes=3, conv1_t_stride=(2,2,1)).to(device)
    # class_weights = torch.tensor([1.0, 10.0, 40.0]).to('cuda')
    loss_func = torch.nn.CrossEntropyLoss()
    lr = args.lr
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr)
    elif args.optimizer == "Adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr)
    
    epochs = args.epochs

    val_loss_min = 100

    for epoch in range(epochs):

        print(f"epoch {epoch + 1}/{epochs}")
        
        epoch_acc, epoch_loss = train_epoch(args, epoch, train_loader, train_len, model, loss_func, optimizer, device)
        print(f"epoch {epoch} average loss: {epoch_loss: .4f}, average acc: {epoch_acc}")

        val_acc, val_loss = val_epoch(args, epoch, val_loader, val_len, model, loss_func, device)
        print(f"epoch {epoch} val_average loss: {val_loss: .4f}, val_average acc: {val_acc}")

        new_best = False

        if val_loss_min > val_loss:
            save_checkpoint(args.log_dir + f"/model_{epoch}.pt", epoch, model)
            print('new best ({:.5f} --> {:.5f}).'.format(val_loss_min, val_loss))
            
            val_loss_min = val_loss
            new_best = True

        if new_best:
            print('Copying to model.pt new best model!!!!')
            shutil.copyfile(os.path.join(args.log_dir, f'model_{epoch}.pt'), os.path.join(args.log_dir, 'model_best.pt'))

from re import L
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def calculate_accuracy(outputs, targets):
    num_correct = 0.0
    metric_count = 0
    
    with torch.no_grad():
        value = torch.eq(outputs.argmax(dim=1), targets.argmax(dim=-1))
        metric_count += len(value)
        num_correct += value.sum().item()

        return num_correct / metric_count

def conf_matrix(args, acc_targets, acc_outputs, epoch, mode):
    with torch.no_grad():
        cm = confusion_matrix(acc_targets, acc_outputs)
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f"{v2}\n{v3}" for v2, v3 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(3,3)
        sns.set(font_scale=0.5)
        disp = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
        disp.plot()
        plt.savefig(args.log_dir + f'/conf_mat_{mode}_{epoch}.png')
        plt.close()

def save_checkpoint(save_file_path, epoch, model):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    save_states = {
        'epoch': epoch,
        'state_dict': model_state_dict,
    }
    torch.save(save_states, save_file_path)

# def print_args(args):
#     print('***********************************************')
#     print('epochs: ', args.epochs)
#     print('log_dir: ', args.log_dir)
#     print('lr: ', args.lr)
#     print('scheduler w/o: ', args.scheduler_switch)
#     print('scheduler: ', args.lr_scheduler)
#     print('Visible GPUs: ', args.cuda_visible_devices)
#     print('optimizer: ', args.optimizer)
#     print('Batchsize: ', args.batch_size)
#     print('***********************************************')


if __name__ == '__main__':
    main()