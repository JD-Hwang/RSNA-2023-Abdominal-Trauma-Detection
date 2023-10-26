import numpy as np
import os
from tqdm.auto import tqdm
import logging
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
# from dataset.classifier_dataset import liverDataset
from sklearn.metrics import accuracy_score, roc_auc_score
import monai
import argparse
import torch.nn as nn
import torch.nn.functional as F
import monai.networks.nets as nets
from metric import *
from data_utils import get_loader
from torchvision.models import resnet50, ResNet50_Weights, densenet121, DenseNet121_Weights
from acsconv.converters import ACSConverter

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
            
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
# function of seed everything
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# function of counting trainable parameters
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

seed_everything()


def get_args_parser():
    parser = argparse.ArgumentParser('liver train script', add_help=False)
    # Dataset parameters  
    parser.add_argument("--epochs", default=300)
    parser.add_argument('--batch-size',  default=8, type=int)
    parser.add_argument('--fold',  default=0, type=int)
    parser.add_argument('--sampling',  default=1, type=int)
    parser.add_argument('--full_res',  default=False, type=str2bool)
    parser.add_argument('--shape', nargs='+', type=int, help='input shape [y, x, z]')
    parser.add_argument('--weight_sample',  default=False, type=str2bool)
    parser.add_argument("--temp", default=0, type=int)
    parser.add_argument("--organ", default="liver", type=str, help='liver, spleen, kidney')

    # Model parameters
    parser.add_argument('--out_channel',  default=3,  type=int, help='ce or bce')    
    parser.add_argument('--in_channel',  default=2,  type=int, help='windowing channels')    
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument("--model", default='resnet_acs_conv', type=str)

    # Continue Training (Resume)
    parser.add_argument('--resume',           default='',  help='resume from checkpoint')  
    parser.add_argument('--gpu_ids', default='0', type=str, help='cuda_visible_devices')
    # Save setting
    parser.add_argument('--checkpoint_dir', default="/workspace/wjj910/clf/results_liver/", type=str, help='path where to save checkpoint or output')
    return parser

def main(args): 
    # initialize metrics objects
    train_acc_liver = MetricsCalculator('multi')
    valid_acc_liver = MetricsCalculator('multi')

    train_loader, val_loader, train_len, val_len = get_loader(args)

    ### model and config ###
    EPOCHS = args.epochs
    N_VAL_EPOCH = 2
    if args.model == 'resnet10':
        model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, num_classes=3, conv1_t_stride=(2,2,1)).to(device)
    elif args.model == 'resnet34':
        model = monai.networks.nets.resnet34(spatial_dims=3, n_input_channels=1, num_classes=3, conv1_t_stride=(2,2,1)).to(device)
    elif args.model == 'resnet50':
        model = monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=1, num_classes=3, conv1_t_stride=(2,2,1)).to(device)
    elif args.model == 'densnet121':
        model = monai.networks.nets.densnet121(spatial_dims=3, n_input_channels=1, num_classes=3, conv1_t_stride=(2,2,1)).to(device)
    elif args.model == 'resnet_acs_conv':
        model_2d = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model = ACSConverter(model_2d)        
        model.fc = nn.Linear(in_features=2048, out_features=3, bias=True)
    elif args.model == 'dense_acs_conv':
        model_2d = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        model = ACSConverter(model_2d)
        model.classifier = nn.Linear(in_features=4096, out_features=3, bias=True)
    elif args.model == 'efficientnet-b0':
        model = monai.networks.nets.EfficientNetBN("efficientnet-b0", spatial_dims=3, in_channels=2, num_classes=3)
    
    count_params(model)

    device = torch.device(f"cuda:{args.gpu_ids}")
    model = model.to(device)

    ### load pretrained model ###
    ckpt_path = os.path.join(args.checkpoint_dir ,'best_model.pth')
    if os.path.exists(ckpt_path):
        print('Loading pretrained model')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print('Loaded pretrained model')
    else:
        print('No pretrained model')
        
    if args.weight_sample:
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    else:
        loss_fn = FocalLoss(alpha=1, gamma=2, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    global_valid_loss = 1

    for epoch in range(1, EPOCHS+1):
        progress_bar = tqdm(train_loader, desc='Training', dynamic_ncols=True)
        model.train()
        epoch_train_loss = []
        for batch_idx, batch_data in enumerate(progress_bar):
            image = batch_data['image_train'].to(device).float()
            label = batch_data['label_train'].to(device).float()
            if '_acs_conv' in args.model:
                image = image.repeat(1,3,1,1,1)
            optimizer.zero_grad()
            output = model(image)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'train_loss': loss.item()})
            epoch_train_loss.append(loss.item())
            train_acc_liver.update(output, label)
        
        scheduler.step()
        train_auc = train_acc_liver.compute_auc()
        print(f'Epoch {epoch} train loss: {np.mean(epoch_train_loss)} train auc: {train_auc}')
        
        if epoch % N_VAL_EPOCH == 0:   
            progress_bar = tqdm(val_loader, desc='validation', dynamic_ncols=True)
            model.eval()
            epoch_valid_loss = []
            acc_targets = []
            acc_outputs = []
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(progress_bar):
                    image = batch_data['image_val'].to(device).float()
                    label = batch_data['label_val'].to(device).float()
                    if '_acs_conv' in args.model:
                        image = image.repeat(1,3,1,1,1)
                    image = image
                    label = label
                    output = model(image)
                    loss = loss_fn(output, label)
                    epoch_valid_loss.append(loss.item())
                    progress_bar.set_postfix({'valid_loss': loss.item()})
                    valid_acc_liver.update(output, label)

                    acc_targets.append(output.argmax(dim=-1).item()) 
                    acc_outputs.append(label.argmax(dim=1).item())

            valid_auc = valid_acc_liver.compute_auc()
            print(f'Epoch {epoch} valid loss: {np.mean(epoch_valid_loss)}, valid auc: {valid_auc}')
            conf_matrix(args, acc_outputs, acc_targets, epoch)

            if np.mean(epoch_valid_loss) < global_valid_loss:
                global_valid_loss = np.mean(epoch_valid_loss)
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f'Epoch{epoch}_{global_valid_loss:.5f}.pth'))
                print('Saved best model')
        
        train_acc_liver.reset()
        valid_acc_liver.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    save_dir = os.path.join(args.checkpoint_dir, f'fold{args.fold}')
    os.makedirs(save_dir, exist_ok=True)
    args.checkpoint_dir = save_dir
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    fh = logging.FileHandler(os.path.join(save_dir, 'train.log'), mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)

    print = logger.info
    
    print(args)
    main(args)
    

    
