import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from scipy import ndimage
import SimpleITK as sitk

import torch
from torch.utils.data import DataLoader
import monai.networks.nets as nets
from monai.inferers import sliding_window_inference

from trauma_dataset3d import TraumaDataset_seg_temp
from multiprocessing import Pool
from threading import Thread
from utils.utils import *

### multi threading ###
def work(target_index):
    target_mask = get_target_index_mask(output, target_index)
    target_mask = target_mask.astype(np.uint8)
    save_root_organ = save_root / index_name[target_index]
    
    # if no target mask, save full image
    if target_mask.sum() == 0:
        print(f"no target mask for {patient_id}_{series_id} in organ {index_name[target_index]}")
        save_image(image_np, save_root_organ / 'image' / f"{patient_id}_{series_id}_fullres.nii.gz")
        return
    
    # save images using multi threading
    image_fullres, image_patch, mask_fullres, mask_patch = get_patch_from_mask(image_np, target_mask)
        
    p1 = Thread(target=save_image, args=(image_fullres, save_root_organ / 'image' / f"{patient_id}_{series_id}_fullres.nii.gz"))
    p2 = Thread(target=save_image, args=(image_patch, save_root_organ / 'image' / f"{patient_id}_{series_id}_patch.nii.gz"))
    p3 = Thread(target=save_image, args=(mask_fullres, save_root_organ / 'mask' / f"{patient_id}_{series_id}_fullres.nii.gz"))
    p4 = Thread(target=save_image, args=(mask_patch, save_root_organ / 'mask' / f"{patient_id}_{series_id}_patch.nii.gz"))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    
if __name__ == "__main__":
        
    device = torch.device("cuda:2")
    
    model = nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(64, 128, 256, 512),
        strides=(2, 2, 2),
        num_res_units=2
    )
    
    model = model.to(device)
    ckpt_path = 'results/unet_seg/best_model0.952.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    index_name = {
        1: "liver",
        2: "spleen",
        3: "kidney",
        4: "bowel"
    }
    
    valid_ds = TraumaDataset_seg_temp(mode="train", fold=0, min_depth=32)
    valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=4)
    save_root = Path('data/dataset_final/ORGANS_UNET_V2/')
    
    progress_bar = tqdm(valid_dl)
    for batch_idx, sample in enumerate(progress_bar):
        path = valid_ds.dicom_dirs[batch_idx]
        patient_id = path.parent.name
        series_id = path.name
        
        image = sample['image'].as_tensor()
    
        with torch.no_grad():
            output = sliding_window_inference(image, 
                                            roi_size=(32,192,192), 
                                            sw_device=device, 
                                            predictor=model, 
                                            sw_batch_size=8,
                                            mode='gaussian')
        
        image_np = image.squeeze().cpu().numpy()
        output_np = output.argmax(dim=1, keepdim=True).squeeze().cpu().numpy()    
        
        target_indexs = list(range(1, 5))
        with Pool(processes=4) as pool:
            list(pool.map(work, target_indexs))
    
    torch.cuda.empty_cache()

    valid_ds = TraumaDataset_seg_temp(mode="valid", fold=0, min_depth=32)
    valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=4)
    save_root = Path('data/dataset_final/ORGANS_UNET_V2/')
    
    progress_bar = tqdm(valid_dl)
    for batch_idx, sample in enumerate(progress_bar):
        path = valid_ds.dicom_dirs[batch_idx]
        patient_id = path.parent.name
        series_id = path.name
        
        image = sample['image'].as_tensor()
    
        with torch.no_grad():
            output = sliding_window_inference(image, 
                                            roi_size=(32,192,192), 
                                            sw_device=device, 
                                            predictor=model, 
                                            sw_batch_size=8,
                                            mode='gaussian')
        
        image_np = image.squeeze().cpu().numpy()
        output_np = output.argmax(dim=1, keepdim=True).squeeze().cpu().numpy()    
        
        target_indexs = list(range(1, 5))
        with Pool(processes=4) as pool:
            list(pool.map(work, target_indexs))