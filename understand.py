# Download weights and data
# git clone https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2.git prj
# pip install gdown && \
#     gdown 1LH9E5D273kOJXrUmBv_s2hXuOZV-dR65 -O weights.zip && \
#     unzip weights.zip && \
#     mv Self-supervised_Anatomical_Embeddings/checkpoints . && \
#     mv Self-supervised_Anatomical_Embeddings/data . && \
#     rm -r Self-supervised_Anatomical_Embeddings weights.zip

# Prepare the data
# ipython misc/lymphnode_preprocess_crop_multi_process.py

from sam.apis.train import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmcv import Config
from sam import *
from torch.utils.data import DataLoader

import torch
from mmcv.parallel import DataContainer

cfg = Config.fromfile("configs/sam/sam_NIHLN.py")
cfg.model

model = build_detector(cfg.model)
ds = build_dataset(cfg.data.train)


def unbox(dc):
    """Extracts the tensor or list from an MMCV DataContainer."""
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

model = model.train().float()


# CodeSpace's memory only support one batch's calculation
for img, meshgrid, valid, meta in dl:
    losses = model.forward_train(
        img=img, img_metas=meta, meshgrid=meshgrid, valid=valid
    )
    print(losses)
    break

# About the MMCV dataset and dataloader
# from mmdet.datasets.pipelines import Compose
# p = Compose(cfg.data.train['pipeline'])


# from mmdet.datasets import build_dataset, build_dataloader

# dl = build_dataloader(ds, 10, 4, 0) # dataset, batch_size, n_workers, n_gpus

# l = list(dl)

# len(l) # 1
# len(l[0]) # 2
# len(l[0][0]) # 4 dict
# l[0][0].keys() # dict_keys(['img_metas', 'img', 'meshgrid', 'valid'])

# l[0][0]['img'].shape # AttributeError: 'DataContainer' object has no attribute 'shape'
# l[0][0]['img'].data # a list of one item
# l[0][0]['img'].data[0].shape # torch.Size([10, 1, 32, 96, 96])

# model(img=l[0][0]['img'].data[0], img_metas=l[0][0]['img_metas'], meshgrid=l[0][0]['meshgrid'].data[0], valid=l[0][0]['valid'].data[0]) # Out[79]: {'loss': tensor(11.3252, grad_fn=<AddBackward0>)}
