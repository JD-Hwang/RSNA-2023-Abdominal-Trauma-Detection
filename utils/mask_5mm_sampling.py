import os
import numpy as np
import json
from pathlib import Path
from natsort import natsorted
from tqdm.auto import tqdm
import pandas as pd
import SimpleITK as sitk
from multiprocessing import Pool

def process_series(series_nifti):
    patient_id = series_nifti.parent.name
    series_id = series_nifti.name[:-7]
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(series_nifti)))

    n_d = len(mask)
    curr_thick = df_dicom.loc[(patient_id, series_id), "SliceThickness"].mean()
    target_thick = 5
    target_n_d = round(n_d * curr_thick / target_thick)
    percentiles = np.linspace(0, 100, target_n_d)
    indexs = np.percentile(np.arange(n_d), percentiles).round(0).astype(int)
    target_mask = mask[indexs]
    save_path = save_root / patient_id / (series_id + '.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(target_mask), str(save_path))

if __name__ == "__main__":
    data_root = Path('data/train_masks_nifti/')
    save_root = Path('data/train_masks_nifti_5mm/')
    df_dicom = pd.read_parquet("data/train_dicom_tags.parquet")
    df_dicom["PatientID"] = df_dicom["PatientID"].astype(str)
    df_dicom["serie"] = df_dicom["SeriesInstanceUID"].apply(lambda x: x.split(".")[-1])
    df_dicom = df_dicom.set_index(["PatientID", "serie"]).sort_index()
    
    patient_dirs = list(data_root.iterdir())
    series_niftis = []
    for patient_dir in patient_dirs:
        series_niftis.extend(list(patient_dir.iterdir()))

    with Pool(processes=os.cpu_count()) as pool:
        list(tqdm(pool.imap(process_series, series_niftis), total=len(series_niftis)))