import pandas as pd
from monai import data, transforms
import natsort
import glob
import numpy as np
import torch
import os
from torch.utils.data import WeightedRandomSampler, DataLoader

#organ = ['kidney', 'liver', 'spleen']

def get_loader(args):
    if args.full_res == True:
        a_SET_0_healthy_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_fullres_label_fold/fold_0/healthy/*.nii.gz"))[:]
        a_SET_0_low_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_fullres_label_fold/fold_0/low/*.nii.gz"))[:]
        a_SET_0_high_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_fullres_label_fold/fold_0/high/*.nii.gz"))[:]
    elif args.full_res == False:
        a_SET_0_healthy_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_patch_label_fold/fold_0/healthy/*.nii.gz"))[:]
        a_SET_0_low_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_patch_label_fold/fold_0/low/*.nii.gz"))[:]
        a_SET_0_high_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_patch_label_fold/fold_0/high/*.nii.gz"))[:]

    if args.full_res == True:
        a_SET_1_healthy_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_fullres_label_fold/fold_1/healthy/*.nii.gz"))[:]
        a_SET_1_low_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_fullres_label_fold/fold_1/low/*.nii.gz"))[:]
        a_SET_1_high_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_fullres_label_fold/fold_1/high/*.nii.gz"))[:]
    elif args.full_res == False:
        a_SET_1_healthy_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_patch_label_fold/fold_1/healthy/*.nii.gz"))[:]
        a_SET_1_low_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_patch_label_fold/fold_1/low/*.nii.gz"))[:]
        a_SET_1_high_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_patch_label_fold/fold_1/high/*.nii.gz"))[:]

    if args.full_res == True:
        a_SET_2_healthy_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_fullres_label_fold/fold_2/healthy/*.nii.gz"))[:]
        a_SET_2_low_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_fullres_label_fold/fold_2/low/*.nii.gz"))[:]
        a_SET_2_high_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_fullres_label_fold/fold_2/high/*.nii.gz"))[:]
    elif args.full_res == False:
        a_SET_2_healthy_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_patch_label_fold/fold_2/healthy/*.nii.gz"))[:]
        a_SET_2_low_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_patch_label_fold/fold_2/low/*.nii.gz"))[:]
        a_SET_2_high_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_patch_label_fold/fold_2/high/*.nii.gz"))[:]

    if args.full_res == True:
        a_SET_3_healthy_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_fullres_label_fold/fold_3/healthy/*.nii.gz"))[:]
        a_SET_3_low_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_fullres_label_fold/fold_3/low/*.nii.gz"))[:]
        a_SET_3_high_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_fullres_label_fold/fold_3/high/*.nii.gz"))[:]
    elif args.full_res == False:
        a_SET_3_healthy_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_patch_label_fold/fold_3/healthy/*.nii.gz"))[:]
        a_SET_3_low_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_patch_label_fold/fold_3/low/*.nii.gz"))[:]
        a_SET_3_high_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_patch_label_fold/fold_3/high/*.nii.gz"))[:]

    if args.full_res == True:
        a_SET_4_healthy_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_fullres_label_fold/fold_4/healthy/*.nii.gz"))[:]
        a_SET_4_low_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_fullres_label_fold/fold_4/low/*.nii.gz"))[:]
        a_SET_4_high_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_fullres_label_fold/fold_4/high/*.nii.gz"))[:]
    elif args.full_res == False:
        a_SET_4_healthy_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_patch_label_fold/fold_4/healthy/*.nii.gz"))[:]
        a_SET_4_low_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_patch_label_fold/fold_4/low/*.nii.gz"))[:]
        a_SET_4_high_path = natsort.natsorted(glob.glob(f"/workspace/dataset_final/ORGANS_UNET_V2/{args.organ}/image_patch_label_fold/fold_4/high/*.nii.gz"))[:]



    if args.fold == 0:
        train_healthy_path = a_SET_1_healthy_path + a_SET_2_healthy_path + a_SET_3_healthy_path + a_SET_4_healthy_path
        train_low_path = a_SET_1_low_path + a_SET_2_low_path + a_SET_3_low_path + a_SET_4_low_path
        train_high_path = a_SET_1_high_path + a_SET_2_high_path + a_SET_3_high_path + a_SET_4_high_path
        
        valid_healthy_path = a_SET_0_healthy_path
        valid_low_path = a_SET_0_low_path
        valid_high_path = a_SET_0_high_path
    
    elif args.fold == 1:
        train_healthy_path = a_SET_0_healthy_path + a_SET_2_healthy_path + a_SET_3_healthy_path + a_SET_4_healthy_path
        train_low_path = a_SET_0_low_path + a_SET_2_low_path + a_SET_3_low_path + a_SET_4_low_path
        train_high_path = a_SET_0_high_path + a_SET_2_high_path + a_SET_3_high_path + a_SET_4_high_path
        
        valid_healthy_path = a_SET_1_healthy_path
        valid_low_path = a_SET_1_low_path
        valid_high_path = a_SET_1_high_path

    elif args.fold == 2:
        train_healthy_path = a_SET_0_healthy_path + a_SET_1_healthy_path + a_SET_3_healthy_path + a_SET_4_healthy_path
        train_low_path = a_SET_0_low_path + a_SET_1_low_path + a_SET_3_low_path + a_SET_4_low_path
        train_high_path = a_SET_0_high_path + a_SET_1_high_path + a_SET_3_high_path + a_SET_4_high_path
        
        valid_healthy_path = a_SET_2_healthy_path
        valid_low_path = a_SET_2_low_path
        valid_high_path = a_SET_2_high_path

    elif args.fold == 3:
        train_healthy_path = a_SET_0_healthy_path + a_SET_1_healthy_path + a_SET_2_healthy_path + a_SET_4_healthy_path
        train_low_path = a_SET_0_low_path + a_SET_1_low_path + a_SET_2_low_path + a_SET_4_low_path
        train_high_path = a_SET_0_high_path + a_SET_1_high_path + a_SET_2_high_path + a_SET_4_high_path
        
        valid_healthy_path = a_SET_3_healthy_path
        valid_low_path = a_SET_3_low_path
        valid_high_path = a_SET_3_high_path

    elif args.fold == 4:
        train_healthy_path = a_SET_0_healthy_path + a_SET_1_healthy_path + a_SET_2_healthy_path + a_SET_3_healthy_path
        train_low_path = a_SET_0_low_path + a_SET_1_low_path + a_SET_2_low_path + a_SET_3_low_path
        train_high_path = a_SET_0_high_path + a_SET_1_high_path + a_SET_2_high_path + a_SET_3_high_path
        
        valid_healthy_path = a_SET_4_healthy_path
        valid_low_path = a_SET_4_low_path
        valid_high_path = a_SET_4_high_path

    if args.sampling == 1:
        train_healthy_path = train_healthy_path[:500] 
        train_low_path =train_low_path[:] 
        train_high_path =train_high_path[:]

        valid_healthy_path = valid_healthy_path[:80] 
        valid_low_path = valid_low_path[:80] 
        valid_high_path = valid_high_path[:]

        if args. temp == 1:
            train_healthy_path = train_healthy_path[:20] 
            train_low_path =train_low_path[:20] 
            train_high_path =train_high_path[:20]

            valid_healthy_path = valid_healthy_path[:20] 
            valid_low_path = valid_low_path[:20] 
            valid_high_path = valid_high_path[:] 

    print(f"tr_len_healthy : {len(train_healthy_path)}")
    print(f"tr_len_low : {len(train_low_path)}")
    print(f"tr_len_high : {len(train_high_path)}")

    print(f"val_len_healthy : {len(valid_healthy_path)}")
    print(f"val_len_low : {len(valid_low_path)}")
    print(f"val_len_high : {len(valid_high_path)}")


    a_train_path = train_healthy_path + train_low_path + train_high_path
    a_valid_path = valid_healthy_path + valid_low_path + valid_high_path
    
    label_list = []
    for i in range(len(train_healthy_path)):
        label_list.append(0)

    for i in range(len(train_low_path)):
        label_list.append(1)

    for i in range(len(train_high_path)):
        label_list.append(2)

    a_label_train = np.array(torch.nn.functional.one_hot(torch.as_tensor((label_list)))).astype(float)

    label_list = []
    for i in range(len(valid_healthy_path)):
        label_list.append(0)

    for i in range(len(valid_low_path)):
        label_list.append(1)

    for i in range(len(valid_high_path)):
        label_list.append(2)

    a_label_valid = np.array(torch.nn.functional.one_hot(torch.as_tensor((label_list)))).astype(float)

    assert(len(train_healthy_path + train_low_path + train_high_path) == len(a_label_train))
    assert(len(valid_healthy_path + valid_low_path + valid_high_path) == len(a_label_valid))

    files_tr = [{"image_train": img_tr, "label_train": label_tr} for img_tr, label_tr in zip(a_train_path, a_label_train)]
    files_val = [{"image_val": img_val, "label_val": label_val} for img_val, label_val in zip(a_valid_path, a_label_valid)]

    if args.organ == 'liver':
        if args.full_res == True:
            tr_transforms = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image_train"]), # 기본적으로 nibabel -> RAS
                    transforms.EnsureChannelFirstd(keys=["image_train"]), # Ensurechannelfirstd로 변경 가능
                    # transforms.ScaleIntensityRanged(keys=["image_train"], a_min=-0.38, a_max=0.22, b_min=-1.0, b_max=1.0, clip=True),# 
                    transforms.ScaleIntensityRanged(keys=["image_train"], a_min=-0.84, a_max=0.76, b_min=-1.0, b_max=1.0, clip=True),# 
                    transforms.Resized(keys=["image_train"], spatial_size=(256,256,64), mode='nearest'),
                    transforms.EnsureTyped(keys=["image_train", "label_train"]),
                    transforms.ToTensord(keys=["image_train", "label_train"]),
                ]
            )

            val_transforms = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image_val"]),
                    transforms.EnsureChannelFirstd(keys=["image_val"]),
                    transforms.Resized(keys=["image_val"], spatial_size=(256,256,64), mode='nearest'),
                    transforms.ScaleIntensityRanged(keys=["image_val"], a_min=-0.84, a_max=0.76, b_min=-1.0, b_max=1.0, clip=True),#fit48
                    transforms.ToTensord(keys=["image_val"]),
                ]
            )
        
        elif args.full_res == False:
            tr_transforms = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image_train"]), # 기본적으로 nibabel -> RAS
                    transforms.EnsureChannelFirstd(keys=["image_train"]), # Ensurechannelfirstd로 변경 가능
                    transforms.EnsureTyped(keys=["image_train", "label_train"]),
                    transforms.ToTensord(keys=["image_train", "label_train"]),
                ]
            )

            val_transforms = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image_val"]),
                    transforms.EnsureChannelFirstd(keys=["image_val"]),
                    # transforms.ScaleIntensityRanged(keys=["image_val"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),#fit48
                    transforms.ToTensord(keys=["image_val"]),
                ]
            )
    
    elif args.organ != 'liver':
        if args.full_res == True:
            tr_transforms = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image_train"]), # 기본적으로 nibabel -> RAS
                    transforms.EnsureChannelFirstd(keys=["image_train"]), # Ensurechannelfirstd로 변경 가능
                    transforms.EnsureTyped(keys=["image_train", "label_train"]),
                    transforms.ToTensord(keys=["image_train", "label_train"]),
                ]
            )

            val_transforms = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image_val"]),
                    transforms.EnsureChannelFirstd(keys=["image_val"]),
                    # transforms.ScaleIntensityRanged(keys=["image_val"], a_min=-160.0, a_max=240.0, b_min=0.0, b_max=1.0, clip=True),#fit48
                    transforms.ToTensord(keys=["image_val"]),
                ]
            )

        elif args.full_res == False:
            tr_transforms = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image_train"]), # 기본적으로 nibabel -> RAS
                    transforms.EnsureChannelFirstd(keys=["image_train"]), # Ensurechannelfirstd로 변경 가능
                    transforms.EnsureTyped(keys=["image_train", "label_train"]),
                    transforms.ToTensord(keys=["image_train", "label_train"]),
                ]
            )

            val_transforms = transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image_val"]),
                    transforms.EnsureChannelFirstd(keys=["image_val"]),
                    # transforms.ScaleIntensityRanged(keys=["image_val"], a_min=-160.0, a_max=240.0, b_min=0.0, b_max=1.0, clip=True),#fit48
                    transforms.ToTensord(keys=["image_val"]),
                ]
            )

    # new_dataset -> Cachenew_dataset
    train_ds = data.CacheDataset(data = files_tr, transform = tr_transforms, cache_rate = 1.0, num_workers = 12)
    val_ds = data.Dataset(data = files_val, transform = val_transforms)# cache_rate = 1.0, num_workers = 4)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        # persistent_workers=True,
    )

    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
        # persistent_workers=True,
    )

    loader = [train_loader, val_loader, len(train_ds), len(val_ds)]
    print("loader is ver(train, val)")

    return loader
'''
                    # transforms.Resized(keys=["image_train"], spatial_size=(256,256,48), mode='nearest'),
                    # transforms.RandFlipd(keys=["image_train"], prob=0.1),
                    # transforms.RandFlipd(keys=["image_train"], prob=0.1),
                    # transforms.RandFlipd(keys=["image_train"], prob=0.1),
                    # transforms.RandRotate90d(keys=["image_train"], prob=0.1, max_k=3),
                    # transforms.RandRotated(keys=["image_train"], prob=0.1, range_x=0.2),
                    # transforms.RandRotated(keys=["image_train"], prob=0.1, range_y=0.2),
                    # transforms.RandRotated(keys=["image_train"], prob=0.1, range_z=0.2),
                    # transforms.RandZoomd(keys=["image_train"], min_zoom=0.8, max_zoom=1.2, prob=0.1),
                    # transforms.RandScaleIntensityd(keys=["image_train"], factors=0.1, prob=0.1),
                    # transforms.RandShiftIntensityd(keys=["image_train"], offsets=0.1, prob=0.1),

                    # transforms.ScaleIntensityRanged(keys=["image_train"], a_min=-45.0, a_max=105.0, b_min=0.0, b_max=1.0, clip=True),#fit48
                    # transforms.RandFlipd(keys=["image_train"], prob=0.1),
                    # transforms.RandFlipd(keys=["image_train"], prob=0.1),
                    # transforms.RandFlipd(keys=["image_train"], prob=0.1),
                    # transforms.RandRotate90d(keys=["image_train"], prob=0.1, max_k=3),
                    # transforms.RandRotated(keys=["image_train"], prob=0.1, range_x=0.2),
                    # transforms.RandRotated(keys=["image_train"], prob=0.1, range_y=0.2),
                    # transforms.RandRotated(keys=["image_train"], prob=0.1, range_z=0.2),
                    # transforms.RandZoomd(keys=["image_train"], min_zoom=0.8, max_zoom=1.2, prob=0.1),
                    # transforms.RandScaleIntensityd(keys=["image_train"], factors=0.1, prob=0.1),
                    # transforms.RandShiftIntensityd(keys=["image_train"], offsets=0.1, prob=0.1),

                    # transforms.ScaleIntensityRanged(keys=["image_train"], a_min=-160.0, a_max=240.0, b_min=0.0, b_max=1.0, clip=True),#fit48
                    # transforms.RandFlipd(keys=["image_train"], prob=0.1),
                    # transforms.RandFlipd(keys=["image_train"], prob=0.1),
                    # transforms.RandFlipd(keys=["image_train"], prob=0.1),
                    # transforms.RandRotate90d(keys=["image_train"], prob=0.1, max_k=3),
                    # transforms.RandRotated(keys=["image_train"], prob=0.1, range_x=0.2),
                    # transforms.RandRotated(keys=["image_train"], prob=0.1, range_y=0.2),
                    # transforms.RandRotated(keys=["image_train"], prob=0.1, range_z=0.2),
                    # transforms.RandZoomd(keys=["image_train"], min_zoom=0.8, max_zoom=1.2, prob=0.1),
                    # transforms.RandScaleIntensityd(keys=["image_train"], factors=0.1, prob=0.1),
                    # transforms.RandShiftIntensityd(keys=["image_train"], offsets=0.1, prob=0.1),

                    # transforms.ScaleIntensityRanged(keys=["image_train"], a_min=-160.0, a_max=240.0, b_min=0.0, b_max=1.0, clip=True),#fit48
                    # transforms.RandFlipd(keys=["image_train"], prob=0.1),
                    # transforms.RandFlipd(keys=["image_train"], prob=0.1),
                    # transforms.RandFlipd(keys=["image_train"], prob=0.1),
                    # transforms.RandRotate90d(keys=["image_train"], prob=0.1, max_k=3),
                    # transforms.RandRotated(keys=["image_train"], prob=0.1, range_x=0.2),
                    # transforms.RandRotated(keys=["image_train"], prob=0.1, range_y=0.2),
                    # transforms.RandRotated(keys=["image_train"], prob=0.1, range_z=0.2),
                    # transforms.RandZoomd(keys=["image_train"], min_zoom=0.8, max_zoom=1.2, prob=0.1),
                    # transforms.RandScaleIntensityd(keys=["image_train"], factors=0.1, prob=0.1),
                    # transforms.RandShiftIntensityd(keys=["image_train"], offsets=0.1, prob=0.1),
'''