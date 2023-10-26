import numpy as np
import os
from tqdm.auto import tqdm
import logging

import torch
from torch.utils.data import DataLoader

import monai.networks.nets as nets
from monai.losses.dice import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

from dataset.trauma_dataset3d import TraumaDataset_seg

# function of seed everything
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
seed_everything()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)

fh = logging.FileHandler('results/unet_seg/train.log', mode='a')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

logger.addHandler(sh)
logger.addHandler(fh)

print = logger.info

train_ds = TraumaDataset_seg(mode="train", fold=0, min_depth=32)
valid_ds = TraumaDataset_seg(mode="valid", fold=0, min_depth=32)

train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=6, collate_fn=train_ds.collate_fn)
valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=4)

### model and config ###
EPOCHS = 200
N_VAL_EPOCH = 3

# function of counting trainable parameters
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = nets.UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=5,
    channels=(64, 128, 256, 512),
    strides=(2, 2, 2),
    num_res_units=2
)
count_params(model)

device = torch.device("cuda:0")
model = model.to(device)

### load pretrained model ###
ckpt_path = 'results/unet_seg/best_model0.921.pth'
if os.path.exists(ckpt_path):
    print('Loading pretrained model')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print('Loaded pretrained model')
else:
    print('No pretrained model')
    
loss_fn = DiceCELoss(
    include_background=False,
    to_onehot_y=True,
    softmax=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

global_valid_dice = 0.921

for epoch in range(1, EPOCHS+1):
    progress_bar = tqdm(train_dl, desc='Training', dynamic_ncols=True)
    model.train()
    epoch_train_loss = []
    for batch_idx, batch_data in enumerate(progress_bar):
        image = batch_data['image']
        label = batch_data['mask']
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix({'loss': loss.item()})
        epoch_train_loss.append(loss.item())
    print(f'Epoch {epoch} train loss: {np.mean(epoch_train_loss)}')
    
    if epoch % N_VAL_EPOCH == 0:   
        progress_bar = tqdm(valid_dl, desc='validation', dynamic_ncols=True)
        model.eval()
        epoch_valid_loss = []
        epoch_valid_dice = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(progress_bar):
                image = batch_data['image']
                label = batch_data['mask']
                output = sliding_window_inference(image, 
                                                  roi_size=(32,192,192), 
                                                  sw_device=device, 
                                                  predictor=model, 
                                                  sw_batch_size=2)
                loss = loss_fn(output, label)
                epoch_valid_loss.append(loss.item())
                metric = DiceMetric(include_background=False, reduction="mean", num_classes=5)
                metric(output.argmax(dim=1, keepdim=True), label)
                epoch_valid_dice.append(metric.aggregate().item())
                progress_bar.set_postfix({'loss': loss.item(), 'dice': metric.aggregate().item()})
        print(f'Epoch {epoch} valid loss: {np.mean(epoch_valid_loss)}')
        print(f'Epoch {epoch} valid dice: {np.mean(epoch_valid_dice)}')
        if np.mean(epoch_valid_dice) > global_valid_dice:
            global_valid_dice = np.mean(epoch_valid_dice)
            torch.save(model.state_dict(), f'results/unet_seg/best_model{global_valid_dice:.3f}.pth')
            print('Saved best model')
                