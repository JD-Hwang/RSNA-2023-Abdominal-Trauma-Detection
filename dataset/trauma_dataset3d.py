import numpy as np
from pathlib import Path
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
import pydicom
from functools import partial
from monai.transforms import *
from utils.utils import *

class TraumaDataset(Dataset):
    def __init__(self, data_root=Path('./data/train_images_5mm'), mode="train", xy_size=512, depth=200):
        self.data_root = data_root
        self.mode = mode
        self.xy_size = xy_size
        self.depth= depth
        csv_path = f'./data/train_{mode}.csv'
        self.data_csv = pd.read_csv(csv_path) 
        patient_ids = list(self.data_csv.patient_id)

        # meta-data
        self.df_dicom = pd.read_parquet("data/dataset_xlsx/train_dicom_tags.parquet")
        self.df_dicom["PatientID"] = self.df_dicom["PatientID"].astype(str)
        self.df_dicom["serie"] = self.df_dicom["SeriesInstanceUID"].apply(lambda x: x.split(".")[-1])
        self.df_dicom = self.df_dicom.set_index(["PatientID", "serie"]).sort_index()
        
        print(f"Loading {mode} datatset ...")
        patient_dirs = list(data_root.iterdir())
        series_dirs = []
        for patient_dir in patient_dirs:
            if int(patient_dir.name) in patient_ids:
                series_dirs.extend(list(patient_dir.iterdir()))
        self.dicom_dirs = series_dirs
        print(f"{mode} dataset have {len(self.dicom_dirs)} ct volumes")
        
    def __len__(self):
        return len(self.dicom_dirs)
    
    def __getitem__(self, idx):
        dicom_dir = self.dicom_dirs[idx]
        patient_id = dicom_dir.parent.name  
        series_id = dicom_dir.name
        
        #### image part ####
        # reverse 적용시: index 0: 몸 아래(골반), index -1: 몸 위(심장)
        dicoms = sorted(dicom_dir.glob('*.dcm'), reverse=True, key=lambda x: int(x.name[:-4]))
        temp_df = self.df_dicom.loc[(patient_id, series_id)].reset_index()
        temp_index = temp_df['ImagePositionPatient'].apply(lambda x: eval(x)[-1]).sort_values().index
        array = list(temp_df.iloc[temp_index]['InstanceNumber'])
        
        if is_reversed(array):
            dicoms = dicoms[::-1]
            
        image = []
        for dicom in dicoms:
            temp = get_array_pydicom(dicom)
            temp = resize(temp, self.xy_size, 3)
            image.append(temp)
        image = np.stack(image, axis=0)
        image = windowing(image, 40, 400)
        image = crop_or_pad(image, self.depth, self.xy_size)
        image = normalize(image)
        
        #### label part ####      
        label = self.data_csv[self.data_csv.patient_id == int(patient_id)].values[0][1:-2]
        bowel = np.argmax(label[:2])
        extravasation = np.argmax(label[2:4])
        kidney = np.argmax(label[4:7])
        liver = np.argmax(label[7:10])
        spleen = np.argmax(label[10:])
        
        #### convert to tensor ####
        image = torch.FloatTensor(image[np.newaxis, ...])
        bowel = torch.tensor(bowel).long()
        extravasation = torch.tensor(extravasation).long()
        kidney = torch.tensor(kidney).long()
        liver = torch.tensor(liver).long()
        spleen = torch.tensor(spleen).long()
        
        return {"image": image,
                "bowel": bowel,
                "extravasation": extravasation,
                "kidney": kidney,
                "liver": liver,
                "spleen": spleen}
        

############### segmentation ###############        
train_transform = Compose(
        [   
            RandAxisFlipd(keys=['image', 'mask'], prob=0.3),
            Lambdad(keys='image', func=partial(windowing, window_center=50, window_width=500)),
            RandCropByLabelClassesd(
                keys=['image', 'mask'], 
                label_key='mask', 
                spatial_size=(32,192,192),
                ratios=[1,3,3,3,3],
                num_classes=5,
                num_samples=2,
                allow_smaller=False,
                allow_missing_keys=True,
                warn=False
            ),
            RandZoomd(keys=['image', 'mask'], prob=0.3, min_zoom=0.7, max_zoom=1.3,
                      mode=('bilinear', 'nearest-exact'), keep_size=True, padding_mode='constant'),
            RandGaussianNoised(keys=['image'], prob=0.3, std=0.05),
            RandShiftIntensityd(keys=['image'], prob=0.3, offsets=0.2),
            ScaleIntensityd(keys=['image'], minv=0, maxv=1),
            Lambdad(keys='image', func=normalize)
        ]
    )

valid_transform = Compose(
    [   
        Lambdad(keys='image', func=partial(windowing, window_center=50, window_width=500)),
        Lambdad(keys='image', func=normalize)
    ]
)

class TraumaDataset_seg(Dataset):
    def __init__(self, data_root=Path('data/dataset_final/train_images_5mm'), 
                 mode="train", fold=0, xy_size=512, min_depth=32):
        self.data_root = data_root
        self.mask_root = Path('data/dataset_final/train_mask_5mm')
        self.mode = mode
        self.xy_size = xy_size
        self.min_depth = min_depth
        csv_path = f'data/dataset_xlsx/{mode}_fold{fold}.csv'
        self.data_csv = pd.read_csv(csv_path) 
        patient_ids = list(self.data_csv.patient_id)
        
        # meta-data
        self.df_dicom = pd.read_parquet("data/dataset_xlsx/train_dicom_tags.parquet")
        self.df_dicom["PatientID"] = self.df_dicom["PatientID"].astype(str)
        self.df_dicom["serie"] = self.df_dicom["SeriesInstanceUID"].apply(lambda x: x.split(".")[-1])
        self.df_dicom = self.df_dicom.set_index(["PatientID", "serie"]).sort_index()
        
        print(f"Loading {mode} fold {fold} datatset ...")        
        patient_dirs = list(data_root.iterdir())
        series_dirs = []
        for patient_dir in patient_dirs:
            if int(patient_dir.name) in patient_ids:
                series_dirs.extend(list(patient_dir.iterdir()))
        self.dicom_dirs = series_dirs
        print(f"{mode} dataset have {len(self.dicom_dirs)} ct volumes")
                
    def __len__(self):
        return len(self.dicom_dirs)
    
    def __getitem__(self, idx):
        dicom_dir = self.dicom_dirs[idx]
        patient_id = dicom_dir.parent.name
        series_id = dicom_dir.name   
        
        #### image part ####
        # reverse 적용시: index 0: 몸 아래(골반), index -1: 몸 위(심장)
        dicoms = sorted(dicom_dir.glob('*.dcm'), reverse=True, key=lambda x: int(x.name[:-4]))
        temp_df = self.df_dicom.loc[(patient_id, series_id)].reset_index()
        temp_index = temp_df['ImagePositionPatient'].apply(lambda x: eval(x)[-1]).sort_values().index
        array = list(temp_df.iloc[temp_index]['InstanceNumber'])
        
        if is_reversed(array):
            dicoms = dicoms[::-1]
        
        image = []
        for dicom in dicoms:
            temp = get_array_pydicom(dicom)
            temp = resize(temp, self.xy_size, 3)
            image.append(temp)
        image = np.stack(image, axis=0)
        
        #### label part ####     
        mask_origin = get_array_sitk(self.mask_root / patient_id / (series_id + '.nii.gz'))
        mask = []
        for i in range(mask_origin.shape[0]):
            temp = resize(mask_origin[i], self.xy_size, 0)
            mask.append(temp)
        mask = np.stack(mask, axis=0)
    
        #### padding ####
        if image.shape[0] < self.min_depth:
            image, _ = pad(image, self.min_depth, self.xy_size)
            mask, _ = pad(mask, self.min_depth, self.xy_size)
        
        #### convert to tensor ####
        image = torch.FloatTensor(image[np.newaxis, ...])
        mask = torch.LongTensor(mask[np.newaxis, ...])
        
        if self.mode == "train":
            data = train_transform({"image": image,'mask': mask})
        else:
            data = valid_transform({"image": image,'mask': mask})
        return data
    
    # batch is list: [{image: tensor, mask: tensor}, {image: tensor, mask: tensor}]
    def collate_fn(self, batch):
        batch_images = []
        batch_masks = []
 
        for i in range(len(batch)):
            for j in range(2):
                batch_images.append(batch[i][j]["image"])
                batch_masks.append(batch[i][j]["mask"])
        
        batch_images = torch.stack(batch_images, dim=0)
        batch_masks = torch.stack(batch_masks, dim=0)
        return {"image": batch_images, "mask": batch_masks}
    
##### for making segmentation mask #####
class TraumaDataset_seg_temp(Dataset):
    def __init__(self, data_root=Path('data/dataset_final/train_images_5mm'), 
                 mode="train", fold=0, xy_size=512, min_depth=32):
        self.data_root = data_root
        self.mask_root = Path('data/dataset_final/train_mask_5mm')
        self.mode = mode
        self.xy_size = xy_size
        self.min_depth = min_depth
        csv_path = f'data/dataset_xlsx/{mode}_fold{fold}.csv'
        self.data_csv = pd.read_csv(csv_path) 
        patient_ids = list(self.data_csv.patient_id)
        
        # meta-data
        self.df_dicom = pd.read_parquet("data/dataset_xlsx/train_dicom_tags.parquet")
        self.df_dicom["PatientID"] = self.df_dicom["PatientID"].astype(str)
        self.df_dicom["serie"] = self.df_dicom["SeriesInstanceUID"].apply(lambda x: x.split(".")[-1])
        self.df_dicom = self.df_dicom.set_index(["PatientID", "serie"]).sort_index()
        
        print(f"Loading {mode} fold {fold} datatset ...")        
        patient_dirs = list(data_root.iterdir())
        series_dirs = []
        for patient_dir in patient_dirs:
            if int(patient_dir.name) in patient_ids:
                series_dirs.extend(list(patient_dir.iterdir()))
        self.dicom_dirs = series_dirs
        print(f"{mode} dataset have {len(self.dicom_dirs)} ct volumes")
                
    def __len__(self):
        return len(self.dicom_dirs)
    
    def __getitem__(self, idx):
        dicom_dir = self.dicom_dirs[idx]
        patient_id = dicom_dir.parent.name
        series_id = dicom_dir.name   
        
        #### image part ####
        # reverse 적용시: index 0: 몸 아래(골반), index -1: 몸 위(심장)
        dicoms = sorted(dicom_dir.glob('*.dcm'), reverse=True, key=lambda x: int(x.name[:-4]))
        temp_df = self.df_dicom.loc[(patient_id, series_id)].reset_index()
        temp_index = temp_df['ImagePositionPatient'].apply(lambda x: eval(x)[-1]).sort_values().index
        array = list(temp_df.iloc[temp_index]['InstanceNumber'])
        
        if is_reversed(array):
            dicoms = dicoms[::-1]
        
        image = []
        for dicom in dicoms:
            temp = get_array_pydicom(dicom)
            temp = resize(temp, self.xy_size, 3)
            image.append(temp)
        image = np.stack(image, axis=0)
        
        #### label part ####     
        mask_origin = get_array_sitk(self.mask_root / patient_id / (series_id + '.nii.gz'))
        mask = []
        for i in range(mask_origin.shape[0]):
            temp = resize(mask_origin[i], self.xy_size, 0)
            mask.append(temp)
        mask = np.stack(mask, axis=0)
    
        #### padding ####
        if image.shape[0] < self.min_depth:
            image, _ = pad(image, self.min_depth, self.xy_size)
            mask, _ = pad(mask, self.min_depth, self.xy_size)
        
        #### convert to tensor ####
        image = torch.FloatTensor(image[np.newaxis, ...])
        mask = torch.LongTensor(mask[np.newaxis, ...])
        
        if self.mode == "train":
            data = valid_transform({"image": image,'mask': mask})
        else:
            data = valid_transform({"image": image,'mask': mask})
        return data
    