import numpy as np
from pathlib import Path
import torch
import SimpleITK as sitk
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from skimage.transform import resize as resize3d
import albumentations as A
import pydicom
from functools import partial
from monai.transforms import *
from utils.utils import *

# train_transform_spleen = Compose(
#         [   
#             Lambdad(keys='image', func=partial(windowing, window_center=40, window_width=150)),
#             RandZoomd(keys=['image', 'mask'], prob=0.3, min_zoom=0.8, max_zoom=1.2,
#                       mode=('bilinear', 'nearest-exact'), keep_size=True, padding_mode='constant'),
#             RandAxisFlipd(keys=['image', 'mask'], prob=0.3),
            
#             RandGaussianNoised(keys=['image'], prob=0.3, std=0.1),
#             RandShiftIntensityd(keys=['image'], prob=0.3, offsets=0.2),
#             ScaleIntensityd(keys=['image'], minv=0, maxv=1),
#             ToTensord(keys=['image', 'mask'], dtype=torch.float32)
#         ]
#     )

# valid_transform_spleen = Compose(
#     [   
#         Lambdad(keys='image', func=partial(windowing, window_center=40, window_width=150)),
#         ScaleIntensityd(keys=['image'], minv=0, maxv=1),
#         ToTensord(keys=['image', 'mask'], dtype=torch.float32)
#     ]
# )
   
train_transform_spleen = Compose(
        [   
            Lambdad(keys='image', func=partial(windowing, window_center=50, window_width=400)),
            RandZoomd(keys=['image', 'mask'], prob=0.3, min_zoom=0.8, max_zoom=1.2,
                      mode=('bilinear', 'nearest-exact'), keep_size=True, padding_mode='constant'),
            RandAxisFlipd(keys=['image', 'mask'], prob=0.3),
            
            RandGaussianNoised(keys=['image'], prob=0.3, std=0.1),
            RandShiftIntensityd(keys=['image'], prob=0.3, offsets=0.2),
            ScaleIntensityd(keys=['image'], minv=0, maxv=1),
            ToTensord(keys=['image', 'mask'], dtype=torch.float32)
        ]
    )

valid_transform_spleen = Compose(
    [   
        Lambdad(keys='image', func=partial(windowing, window_center=50, window_width=400)),
        ScaleIntensityd(keys=['image'], minv=0, maxv=1),
        ToTensord(keys=['image', 'mask'], dtype=torch.float32)
    ]
)
    
class SpleenDataset(Dataset):
    def __init__(self, data_root=Path('data/dataset_final/ORGANS_UNET_V2/spleen/image/'), 
                 mode="train", fold=0, y_size=256, x_size=256, depth=32, full_res=True):
        self.data_root = data_root
        self.mask_root = Path('data/dataset_final/ORGANS_UNET_V2/spleen/mask/')
        self.mode = mode
        self.full_res = full_res
        self.y_size = y_size
        self.x_size = x_size
        self.depth = depth
        csv_path = f'data/dataset_xlsx/{mode}_fold{fold}.csv'
        self.data_csv = pd.read_csv(csv_path) 
        patient_ids = list(self.data_csv.patient_id)
        
        # meta-data
        self.df_dicom = pd.read_parquet("data/dataset_xlsx/train_dicom_tags.parquet")
        self.df_dicom["PatientID"] = self.df_dicom["PatientID"].astype(str)
        self.df_dicom["serie"] = self.df_dicom["SeriesInstanceUID"].apply(lambda x: x.split(".")[-1])
        self.df_dicom = self.df_dicom.set_index(["PatientID", "serie"]).sort_index()
        
        print(f"Loading {mode} fold {fold} datatset ...")
        if full_res:        
            self.image_paths = [i for i in data_root.iterdir() if i.name.endswith('fullres.nii.gz') and int(i.name.split('_')[0]) in patient_ids]
        else:
            self.image_paths = [i for i in data_root.iterdir() if i.name.endswith('patch.nii.gz') and int(i.name.split('_')[0]) in patient_ids]
            
        patient_ids = [int(i.name.split('_')[0]) for i in self.image_paths]
        self.labels = [self.data_csv[self.data_csv.patient_id == int(i)].values[0][1:-2][10:].argmax() for i in patient_ids]
        self.class_weights = 1 / (np.bincount(self.labels) / np.sum(np.bincount(self.labels)))
        self.sample_weights = self.class_weights[self.labels]
        print(f"{mode} dataset have {len(self.image_paths)} ct volumes")
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_root / image_path.name
        patient_id = image_path.name.split('_')[0]
        series_id = image_path.name.split('_')[1]
                
        image= get_array_sitk(image_path).astype(np.float32) #[-1, 1] to original [-200, 300]
        image = denormlize(image) * 500 -200
        mask = get_array_sitk(mask_path).astype(np.uint8)
        
        # resize x,y axis
        image = resize3d(image, (image.shape[0], self.y_size, self.x_size), order=3)
        mask = resize3d(mask, (mask.shape[0], self.y_size, self.x_size), order=0)
        
        # resize z axis
        image = resize3d(image, (self.depth, self.y_size, self.x_size), order=0)
        mask = resize3d(mask, (self.depth, self.y_size, self.x_size), order=0)
        
        #### Apply transform ####
        image = torch.FloatTensor(image[np.newaxis, ...])
        mask = torch.FloatTensor(mask[np.newaxis, ...])
        
        data = {"image": image,'mask': mask}
        if self.mode == "train":
            data = train_transform_spleen(data)
        else:
            data = valid_transform_spleen(data)
            
        image = torch.cat([data['image'].as_tensor(), data['mask'].as_tensor()], dim=0)
        
        #### label part ####     
        label = self.labels[idx]
        label = torch.tensor(label).long()
        
        if self.mode == "train":
            data = {"image": image,'label':label}
        else:
            data = {"image": image,'label':label}
        return data
    
    
train_transform_cls = Compose(
        [   
            Lambdad(keys='image', func=partial(windowing, window_center=50, window_width=400)),
            RandZoomd(keys=['image', 'mask'], prob=0.3, min_zoom=0.8, max_zoom=1.2,
                      mode=('bilinear', 'nearest-exact'), keep_size=True, padding_mode='constant'),
            RandAxisFlipd(keys=['image', 'mask'], prob=0.3),
            
            RandGaussianNoised(keys=['image'], prob=0.3, std=0.1),
            RandShiftIntensityd(keys=['image'], prob=0.3, offsets=0.2),
            ScaleIntensityd(keys=['image'], minv=0, maxv=1),
            ToTensord(keys=['image', 'mask'], dtype=torch.float32)
        ]
    )

valid_transform_cls = Compose(
    [   
        Lambdad(keys='image', func=partial(windowing, window_center=50, window_width=400)),
        ScaleIntensityd(keys=['image'], minv=0, maxv=1),
        ToTensord(keys=['image', 'mask'], dtype=torch.float32)
    ]
)

class OrganDataset(Dataset):
    def __init__(self, data_root=Path('data/dataset_final/ORGANS_UNET_V2/organ/image/'), 
                 mode="train", fold=0, y_size=256, x_size=256, depth=68, full_res=True):
        self.data_root = data_root
        self.mask_root = Path('data/dataset_final/ORGANS_UNET_V2/organ/mask/')
        self.mode = mode
        self.full_res = full_res
        self.y_size = y_size
        self.x_size = x_size
        self.depth = depth
        csv_path = f'data/dataset_xlsx/{mode}_fold{fold}.csv'
        self.data_csv = pd.read_csv(csv_path) 
        patient_ids = list(self.data_csv.patient_id)
        
        # meta-data
        self.df_dicom = pd.read_parquet("data/dataset_xlsx/train_dicom_tags.parquet")
        self.df_dicom["PatientID"] = self.df_dicom["PatientID"].astype(str)
        self.df_dicom["serie"] = self.df_dicom["SeriesInstanceUID"].apply(lambda x: x.split(".")[-1])
        self.df_dicom = self.df_dicom.set_index(["PatientID", "serie"]).sort_index()
        
        print(f"Loading {mode} fold {fold} datatset ...")
        if full_res:        
            self.image_paths = [i for i in data_root.iterdir() if i.name.endswith('fullres.nii.gz') and int(i.name.split('_')[0]) in patient_ids]
        else:
            self.image_paths = [i for i in data_root.iterdir() if i.name.endswith('patch.nii.gz') and int(i.name.split('_')[0]) in patient_ids]
            
        patient_ids = [int(i.name.split('_')[0]) for i in self.image_paths]
        
        self.kidney_labels = [self.data_csv[self.data_csv.patient_id == int(i)].values[0][1:-2][4:7].argmax() for i in patient_ids]
        self.liver_labels = [self.data_csv[self.data_csv.patient_id == int(i)].values[0][1:-2][7:10].argmax() for i in patient_ids]
        self.spleen_labels = [self.data_csv[self.data_csv.patient_id == int(i)].values[0][1:-2][10:].argmax() for i in patient_ids]
        
        print(f"{mode} dataset have {len(self.image_paths)} ct volumes")
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_root / image_path.name
        patient_id = image_path.name.split('_')[0]
        series_id = image_path.name.split('_')[1]
                
        image= get_array_sitk(image_path).astype(np.float32) 
        image = denormlize(image) * 500 - 200                # [-1, 1] to original [-200, 300]
        mask = get_array_sitk(mask_path).astype(np.uint8)
        
        # resize x,y axis
        image = resize3d(image, (image.shape[0], self.y_size, self.x_size), order=3)
        mask = resize3d(mask, (mask.shape[0], self.y_size, self.x_size), order=0)
        
        # resize z axis
        image = resize3d(image, (self.depth, self.y_size, self.x_size), order=0)
        mask = resize3d(mask, (self.depth, self.y_size, self.x_size), order=0)
        
        #### Apply transform ####
        image = torch.FloatTensor(image[np.newaxis, ...])
        mask = torch.FloatTensor(mask[np.newaxis, ...])
        
        data = {"image": image,'mask': mask}
        if self.mode == "train":
            data = train_transform_cls(data)
        else:
            data = valid_transform_cls(data)
            
        image = torch.cat([data['image'].as_tensor(), data['mask'].as_tensor()], dim=0)
        
        #### label part ####     
        kidney = self.kidney_labels[idx]
        kidney = torch.tensor(kidney).long()
        liver = self.liver_labels[idx]
        liver = torch.tensor(liver).long()
        spleen = self.spleen_labels[idx]
        spleen = torch.tensor(spleen).long()
        
        if self.mode == "train":
            data = {"image":image, 'kidney':kidney, 'liver':liver, 'spleen':spleen}
        else:
            data = {"image":image, 'kidney':kidney, 'liver':liver, 'spleen':spleen}
            
        return data