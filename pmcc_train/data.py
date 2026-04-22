from sam.datasets.dataset3dsam import Dataset3dsam
from torch.utils.data import DataLoader
from mmcv.parallel.data_container import DataContainer
import torch


# 5 ID, 10 images used for fine-tuning
# PMCC_ReIrrad_L03  PMCC_ReIrrad_L06  PMCC_ReIrrad_L24  PMCC_ReIrrad_L25  PMCC_ReIrrad_L27

ds = Dataset3dsam(
    data_dir="data",  # Data folder
    index_file="data/filelist.txt",  # File list (note that don't include extra empty lines)
    pipeline=[
        {"type": "LoadTioImage"},
        {"type": "CropBackground"},
        {"type": "ComputeAugParam_sample"},
        {
            "type": "MultiBranch",
            "view1": [
                {"type": "ExtraAttrs", "tag": "view1"},
                {"type": "Crop"},
                {"type": "Resample"},
                {"type": "Crop", "switch": "fix"},
                {"type": "RescaleIntensity"},
                {"type": "RandomNoise3d"},
                {"type": "GenerateMeshGrid"},
                {"type": "GenerateMetaInfo"},
                {"type": "DefaultFormatBundle3d"},
                {
                    "type": "Collect3d",
                    "keys": ["img", "meshgrid", "valid"],
                    "meta_keys": ("filename", "tag", "crop_info"),
                },
            ],
            "view2": [
                {"type": "ExtraAttrs", "tag": "view2"},
                {"type": "Crop"},
                {"type": "Resample"},
                {"type": "Crop", "switch": "fix"},
                {"type": "RescaleIntensity"},
                {"type": "RandomNoise3d"},
                {"type": "GenerateMeshGrid"},
                {"type": "GenerateMetaInfo"},
                {"type": "DefaultFormatBundle3d"},
                {
                    "type": "Collect3d",
                    "keys": ["img", "meshgrid", "valid"],
                    "meta_keys": ("filename", "tag", "crop_info"),
                },
            ],
        },
    ],
)


def unbox(dc):
    return dc.data if isinstance(dc, DataContainer) else dc


def collate(batch):
    # Two views corresponds to the odd/even channel
    # sam.py line 146
    # view_1_fine = fine_feat[0, :, :, :, :].view(128, -1)
    # view_2_fine = fine_feat[1, :, :, :, :].view(128, -1)

    batch_img = []
    batch_meshgrid = []
    batch_metas = []
    batch_valid = []

    for data in batch:
        # 1. Get two samples from the dataset
        data1 = data[0]  # view 1
        data2 = data[1]  # view 2

        # 2. Manually Batch (Stack) the tensors
        # We combine sample 1 and sample 2 into a batch of size 2
        batch_img += [unbox(data1["img"]), unbox(data2["img"])]
        batch_meshgrid += [unbox(data1["meshgrid"]), unbox(data2["meshgrid"])]
        batch_valid += [unbox(data1["valid"]), unbox(data2["valid"])]

        # 3. Handle img_metas (It must be a list of dicts)
        # unbox(data1['img_metas']) usually returns a single dict
        batch_metas += [unbox(data1["img_metas"]), unbox(data2["img_metas"])]

    return (
        torch.stack(batch_img),
        torch.stack(batch_meshgrid),
        torch.stack(batch_valid),
        batch_metas,
    )


dl = DataLoader(ds, batch_size=1, collate_fn=collate)
