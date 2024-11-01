# Convert mask to .nrrd

import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

data_dir = Path(r'P:\yusun\self-supervised-anatomical-embedding-v2\data\lung')

for f in tqdm(data_dir.iterdir()):
    f_dir, m_dir = f/'1', f/'2'

    f_mask = sitk.ReadImage((f_dir/'mask-lung-dilated.nii.gz').as_posix(), outputPixelType=sitk.sitkUInt8)
    m_mask = sitk.ReadImage((m_dir/'mask-lung-dilated.nii.gz').as_posix(), outputPixelType=sitk.sitkUInt8)

    sitk.WriteImage(f_mask, (f_dir/'mask-lung-dilated.nrrd').as_posix())
    sitk.WriteImage(m_mask, (m_dir/'mask-lung-dilated.nrrd').as_posix())
