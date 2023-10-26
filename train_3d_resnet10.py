import pandas as pd
import numpy as np
import os, shutil
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
import monai.networks.nets as nets
from dataset.trauma_dataset3d import TraumaDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.metrics import MetricsCalculator

# initialize metrics objects
train_acc_bowel = MetricsCalculator('binary')
train_acc_extravasation = MetricsCalculator('binary')
train_acc_liver = MetricsCalculator('multi')
train_acc_kidney = MetricsCalculator('multi')
train_acc_spleen = MetricsCalculator('multi')

val_acc_bowel = MetricsCalculator('binary')
val_acc_extravasation = MetricsCalculator('binary')
val_acc_liver = MetricsCalculator('multi')
val_acc_kidney = MetricsCalculator('multi')
val_acc_spleen = MetricsCalculator('multi')

class MyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.resnet = nets.resnet10(spatial_dims=3, n_input_channels=1, num_classes=1)
        self.resnet.fc = None
        self.active_head = nn.Linear(in_features=512, out_features=2, bias=True)
        self.bowel_head = nn.Linear(in_features=512, out_features=2, bias=True)
        self.kidney_head = nn.Linear(in_features=512, out_features=3, bias=True)
        self.liver_head = nn.Linear(in_features=512, out_features=3, bias=True)
        self.spleen_head = nn.Linear(in_features=512, out_features=3, bias=True)
    
    def forward(self, x):
        x = self.resnet(x)
        active = self.active_head(x)
        bowel = self.bowel_head(x)
        kidney = self.kidney_head(x)
        liver = self.liver_head(x)
        spleen = self.spleen_head(x)
        return active, bowel, kidney, liver, spleen
    
NUM_EPOCHS = 200

device = torch.device("cuda:1")
train_ds = TraumaDataset(xy_size=128, depth=192, mode="train")
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
val_ds = TraumaDataset(xy_size=128, depth=192, mode="valid")
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

# load pretrained model
ckpt_path = 'results/resnet10_3d/resnet_2.082.pth'
if os.path.exists(ckpt_path):
    model = torch.load(ckpt_path, map_location=device)
    model.to(device)
    print("Model Loaded")
else:
    model = MyModel().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay=1e-5, amsgrad=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
loss_fn = nn.CrossEntropyLoss(label_smoothing = 0.05)

prev_val_best_loss = float(2.08)

for epoch in range(NUM_EPOCHS):
    # training
    model.train()
    train_loss = 0.0
    val_loss = 0.0
    
    print(f'Epoch: [{epoch+1}/{NUM_EPOCHS}]')
    
    for batch_idx, batch_data in enumerate(tqdm(train_loader, desc='Training', dynamic_ncols=True)):
        optimizer.zero_grad()
        inputs = batch_data['image'].to(device)
        bowel = batch_data['bowel'].to(device)
        extravasation = batch_data['extravasation'].to(device)
        liver = batch_data['liver'].to(device)
        kidney = batch_data['kidney'].to(device)
        spleen = batch_data['spleen'].to(device)
        
        e, b, k, l, s = model(inputs)
        b_loss = loss_fn(b, bowel)
        e_loss = loss_fn(e, extravasation)
        l_loss = loss_fn(l, liver)
        k_loss = loss_fn(k, kidney)
        s_loss = loss_fn(s, spleen)
        
        total_loss = b_loss + e_loss + l_loss + k_loss + s_loss
        total_loss.backward()
        optimizer.step()
        
        # calculate training metrics
        train_loss += total_loss.item()
        train_acc_bowel.update(b, bowel)
        train_acc_extravasation.update(e, extravasation)
        train_acc_liver.update(l, liver)
        train_acc_kidney.update(k, kidney)
        train_acc_spleen.update(s, spleen)
        
    train_loss = train_loss/(batch_idx+1)
    
    # validation
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        for batch_idx, batch_data in enumerate(tqdm(val_loader, desc='Validation', dynamic_ncols=True)):
                                                    
            inputs = batch_data['image'].to(device)
            bowel = batch_data['bowel'].to(device)
            extravasation = batch_data['extravasation'].to(device)
            liver = batch_data['liver'].to(device)
            kidney = batch_data['kidney'].to(device)
            spleen = batch_data['spleen'].to(device)
    
            e, b, k, l, s = model(inputs)
            b_loss = loss_fn(b, bowel)
            e_loss = loss_fn(e, extravasation)
            l_loss = loss_fn(l, liver)
            k_loss = loss_fn(k, kidney)
            s_loss = loss_fn(s, spleen)
            
            total_loss = b_loss + e_loss + l_loss + k_loss + s_loss
            
            # calculate validation metrics
            val_loss += total_loss.item()
            val_acc_bowel.update(b, bowel)
            val_acc_extravasation.update(e, extravasation)
            val_acc_liver.update(l, liver)
            val_acc_kidney.update(k, kidney)
            val_acc_spleen.update(s, spleen)
        
        val_loss = val_loss/(batch_idx+1)
        scheduler.step(val_loss)
    
    if val_loss < prev_val_best_loss:
        prev_val_best_loss = val_loss
        print("Validation Loss improved, Saving Model...")
        torch.save(model, f'results/resnet10_3d/resnet_{val_loss:.3f}.pth')
    
    
    # verbose
    print('\n<====Train Accuracy====>')
    print(f'Bowel: {train_acc_bowel.compute_accuracy()}')
    print(f'Extravasation: {train_acc_extravasation.compute_accuracy()}')
    print(f'Liver: {train_acc_liver.compute_accuracy()}')
    print(f'Kidney: {train_acc_kidney.compute_accuracy()}')
    print(f'Spleen: {train_acc_spleen.compute_accuracy()}')
    
    print('\n<====Val Accuracy====>')
    print(f'Bowel: {val_acc_bowel.compute_accuracy()}')
    print(f'Extravasation: {val_acc_extravasation.compute_accuracy()}')
    print(f'Liver: {val_acc_liver.compute_accuracy()}')
    print(f'Kidney: {val_acc_kidney.compute_accuracy()}')
    print(f'Spleen: {val_acc_spleen.compute_accuracy()}')
    
    print('\n<====Train AUC====>')
    print(f'Bowel: {train_acc_bowel.compute_auc()}')
    print(f'Extravasation: {train_acc_extravasation.compute_auc()}')
    print(f'Liver: {train_acc_liver.compute_auc()}')
    print(f'Kidney: {train_acc_kidney.compute_auc()}')
    print(f'Spleen: {train_acc_spleen.compute_auc()}')
    
    print('\n<====Val AUC====>')
    print(f'Bowel: {val_acc_bowel.compute_auc()}')
    print(f'Extravasation: {val_acc_extravasation.compute_auc()}')
    print(f'Liver: {val_acc_liver.compute_auc()}')
    print(f'Kidney: {val_acc_kidney.compute_auc()}')
    print(f'Spleen: {val_acc_spleen.compute_auc()}')
    
    print(f'\nMean Train Loss: {train_loss:.3f}')
    print(f'Mean Val Loss: {val_loss:.3f}\n')
    
    #reset metrics
    train_acc_bowel.reset()
    train_acc_extravasation.reset()
    train_acc_liver.reset()
    train_acc_kidney.reset()
    train_acc_spleen.reset()
    val_acc_bowel.reset()
    val_acc_extravasation.reset()
    val_acc_liver.reset()
    val_acc_kidney.reset()
    val_acc_spleen.reset()