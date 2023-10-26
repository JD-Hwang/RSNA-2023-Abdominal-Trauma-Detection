import os, shutil, random
import numpy as np
import json
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import multiprocessing as mp
from functools import partial

data_root = Path('data/train_images/')
save_root = Path('data/train_images_5mm/')

# meta-data
df_dicom = pd.read_parquet("data/train_dicom_tags.parquet")
# Gest series folder
df_dicom["PatientID"] = df_dicom["PatientID"].astype(str)
df_dicom["serie"] = df_dicom["SeriesInstanceUID"].apply(lambda x: x.split(".")[-1])
df_dicom = df_dicom.set_index(["PatientID", "serie"]).sort_index()

# get all patient/series directories
patient_dirs = list(data_root.iterdir())
series_dirs = []
for patient_dir in patient_dirs:
    series_dirs.extend(list(patient_dir.iterdir()))
    
#### mp.Pool version ####
def sampling_5mm(series_dir, df_dicom):
    patient_id = series_dir.parent.name
    series_id = series_dir.name
    src_dicoms = sorted(series_dir.glob('*.dcm'), key=lambda x: int(x.name[:-4]), reverse=True)
    n_d = len(src_dicoms)
    curr_thick = df_dicom.loc[(patient_id, series_id), "SliceThickness"].mean()
    target_thick = 5
    target_n_d = round(n_d * curr_thick / target_thick)
    percentiles = np.linspace(0, 100, target_n_d)
    indexs = np.percentile(np.arange(n_d), percentiles).round(0).astype(int)
    
    save_dicoms = []
    for index in list(indexs):
        save_dicoms.append(src_dicoms[index])
        
    for i in range(len(save_dicoms)):
        dicom = save_dicoms[i]
        dst = save_root / dicom.parent.parent.name / dicom.parent.name / dicom.name
        if not dst.parent.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(dicom, dst)
        
if __name__ == '__main__':
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(partial(sampling_5mm, df_dicom=df_dicom), series_dirs), total=len(series_dirs)):
            pass
        

    