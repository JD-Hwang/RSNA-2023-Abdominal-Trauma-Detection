import numpy as np
import torch
import SimpleITK as sitk
import albumentations as A
import pydicom
from scipy import ndimage

def save_image(image, save_path):
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(image), str(save_path))
    
def is_reversed(array):
    if len(array) > 1:
        if int(array[0]) < int(array[1]):
            return True
        else:
            return False
    else:
        return False
    
def get_array_sitk(path):
    array = np.array(sitk.GetArrayFromImage(sitk.ReadImage(path)).squeeze(), dtype=np.float32)
    return array

def get_array_pydicom(path):
    ds = pydicom.dcmread(path)
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = ds.pixel_array
    if ds.PixelRepresentation == 1:
        bit_shift = ds.BitsAllocated - ds.BitsStored
        dtype = pixel_array.dtype 
        pixel_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
    intercept = float(ds.RescaleIntercept)
    slope = float(ds.RescaleSlope)
    pixel_array = (pixel_array * slope) + intercept
    pixel_array = np.array(pixel_array, dtype=np.float32)
    return pixel_array

# normalize [0, 1] to [-1, 1]
def normalize(image):
    if isinstance(image, np.ndarray):
        return np.clip((image - 0.5)*2., -1., 1.)
    elif isinstance(image, torch.Tensor):
        return torch.clamp((image - 0.5)*2., -1., 1.)
    
# denormalize [-1, 1] to [0, 1]
def denormlize(image):
    if isinstance(image, np.ndarray):
        return np.clip((image + 1.)/2., 0., 1.)
    elif isinstance(image, torch.Tensor):
        return torch.clamp((image + 1.)/2., 0., 1.)

# windowing function
# original data range [-1024, 3071] 
# general abdomen window [40, 400]
def windowing(img, window_center=40, window_width=400):
    lower_bound = window_center - window_width/2
    upper_bound = window_center + window_width/2
    img = (np.clip(img, lower_bound, upper_bound) - lower_bound) / window_width
    return img

def resize(image, image_size=512, order=1):
        return A.resize(image, image_size, image_size, order)
    
# To make all images have same depth
def crop_or_pad(x, depth=200, image_size=512):
    target = np.zeros((depth, image_size, image_size))
    
    # crop: 장기 아래 부분을 제거 
    if x.shape[0] > depth:
        start_idx = x.shape[0] - depth
        target = x[start_idx:start_idx+depth, ...]
    # pad: 장기 아래 부분을 0으로 채움
    elif x.shape[0] < depth:
        start_idx = depth - x.shape[0]
        target[start_idx:start_idx+x.shape[0], ...] = x
    else:
        target = x
    return target 

def pad(x, min_depth=32, image_size=512):
    if len(x.shape) == 5 and isinstance(x, torch.Tensor):
        x = x.squeeze().numpy()
        original_depth = x.shape[0]
        target = torch.zeros((min_depth, image_size, image_size), device=x.device)
        assert original_depth < min_depth
        start_idx = min_depth - original_depth
        target[start_idx:start_idx+x.shape[0], ...] = x
        target = target.unsqueeze(0).unsqueeze(0)
    elif len(x.shape) == 3 and isinstance(x, np.ndarray):
        original_depth = x.shape[0]
        target = np.zeros((min_depth, image_size, image_size))
        assert original_depth < min_depth
        start_idx = min_depth - original_depth
        target[start_idx:start_idx+x.shape[0], ...] = x
    else:
        raise ValueError("Invalid input type")
    return target, original_depth

def reverse_pad(x, original_depth=20, min_depth=32):
    if len(x.shape) == 5 and isinstance(x, torch.Tensor):
        x = x.squeeze().numpy()
        assert x.shape[0] == min_depth
        start_idx = min_depth - original_depth
        x = x[start_idx:start_idx+original_depth, ...]
        x = x.unsqueeze(0).unsqueeze(0)
    elif len(x.shape) == 3 and isinstance(x, np.ndarray):
        assert x.shape[0] == min_depth
        start_idx = min_depth - original_depth
        x = x[start_idx:start_idx+original_depth, ...]
    else:
        raise ValueError("Invalid input type")
    return x

# 1: liver, 2: spleen, 3: kidney, 4: bowel
def get_target_index_mask(mask, target_index=1):
    """
    mask: (1, out_channel, depth, height, width) tensor
    """
    temp = mask.clone()
    if temp.shape[1] != 1:
        temp = temp.argmax(dim=1, keepdim=True)
    # ret shape: (depth, height, width)
    ret = temp.squeeze().detach().cpu().numpy()
    ret = (ret.copy() == target_index).astype(np.uint8)
    # get largest connected component: liver, spleen 
    if target_index == 1 or target_index==2:
        ret = get_largest_connected_component(ret)
    else:
        ret = get_above_threshold_component(ret, threshold=0.1) # kidney, bowel
    return ret

def get_patch_from_mask(image_np, mask_np):
    """
    mask_np: (depth, height, width) ndarray
    """
    z = np.where(mask_np.sum(1).sum(1) > 0)[0]  # depth
    y = np.where(mask_np.sum(0).sum(1) > 0)[0]  # height
    x = np.where(mask_np.sum(0).sum(0) > 0)[0]  # width
    
    x1, x2 = max(0, x[0] - 5), min(mask_np.shape[1], x[-1] + 5)
    y1, y2 = max(0, y[0] - 5), min(mask_np.shape[2], y[-1] + 5)
    z1, z2 = max(0, z[0] - 3), min(mask_np.shape[0], z[-1] + 3)
    
    ret_image_fullres = image_np[z1:z2, ...]
    ret_image = image_np[z1:z2, y1:y2, x1:x2]
    ret_mask_fullres = mask_np[z1:z2, ...]
    ret_mask = mask_np[z1:z2, y1:y2, x1:x2]
    return ret_image_fullres, ret_image, ret_mask_fullres, ret_mask

def get_largest_connected_component(mask):
    img_labels, num_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, img_labels, range(num_labels+1))
    max_index = sizes.argmax()
    mask = (img_labels == max_index).astype(np.uint8)
    return mask

def get_above_threshold_component(mask, threshold=0.1):
    img_labels, num_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, img_labels, range(num_labels+1))
    # sizes to percentage
    sizes = sizes / (sizes.sum() + 1e-8)
    # remain only above 10% of the label
    mask_size = sizes < threshold
    remove_pixel = mask_size[img_labels]
    img_labels[remove_pixel] = 0
    mask[img_labels == 0] = 0
    return mask