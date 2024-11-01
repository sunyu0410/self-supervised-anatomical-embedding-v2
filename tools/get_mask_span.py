import SimpleITK as sitk
import numpy as np

def get_mask_z_span(filepath):
    mask = sitk.ReadImage(filepath)
    arr = sitk.GetArrayFromImage(mask)
    z, y, x = np.where(arr!=0)
    spans = [(i.min(), i.max()) for i in (x, y, z)]
    return dict(zip('zyx', spans))

if __name__ == "__main__":
    # filepath = r'p:\yusun\self-supervised-anatomical-embedding-v2\data\lung\PMCC_ReIrrad_14\1\mask-lung-dilated.nrrd'
    import sys
    filepath = sys.argv[1]
    print(get_mask_z_span(filepath))