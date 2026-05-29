# After 500 epochs of fine tune
# run the inference on the non-ft images

# ipython

import sys

sys.path.append("tools")

from pathlib import Path
from tools.matcher import Match, init
from tools.pmcc_rmse import get_points, Mapper
import SimpleITK as sitk
import numpy as np

from interfaces import init, get_embedding, get_sim_embed_loc, normalize
from utils import read_image, visualize

from tqdm import tqdm
from pathlib import Path

import pandas as pd

def find_corresp(model, pt1_list, im1_file, im2_file):
    im1, normed_im1, norm_info_1 = read_image(im1_file, is_MRI=False)
    im2, normed_im2, norm_info_2 = read_image(im2_file, is_MRI=False)

    emb1 = get_embedding(normed_im1, model)
    emb2 = get_embedding(normed_im2, model)
    
    emb1[0] = emb1[0].float()
    emb1[1] = emb1[1].float()
    emb2[0] = emb2[0].float()
    emb2[1] = emb2[1].float()

    pt2_list = []
    for pt1 in tqdm(pt1_list):
        pt1_normed = np.array(pt1) * norm_info_1
        pt2_normed, score = get_sim_embed_loc(
            emb1,
            emb2,
            pt1_normed,
            (im2["shape"][3], im2["shape"][1], im2["shape"][2]),
            norm_info=norm_info_2,
            write_sim=False,
            use_sim_coarse=True,
        )
        pt2 = np.array(pt2_normed).astype(int)
        pt2_list.append(pt2)

    return pt2_list


config_file = "configs/sam/sam_NIHLN.py"
ft_ids = ['PMCC_ReIrrad_L03',  'PMCC_ReIrrad_L06',  'PMCC_ReIrrad_L24',  'PMCC_ReIrrad_L25',  'PMCC_ReIrrad_L27']
data_dir = Path('data')


for e in range(15):

    checkpoint_file = f"pmcc_train/checkpoints_lr1e-7/sam-v1-ft-epoch-{e}.pth"
    model = init(config_file, checkpoint_file)

    for f in data_dir.iterdir():

        if not f.name.startswith('PMCC'): continue
        if f.name in ft_ids: continue

        outfile = Path(f'pmcc_train/after_ft_lr1e-7/result_epoch_{e}_{f.name}.pkl')
        if outfile.exists(): continue

        print(f)

        im1_file = str(f / "c1.nii.gz")

        if f.name == 'PMCC_ReIrrad_L14':
            # This patient was cropped for registration to reduce c2
            # When saved using 3D Slicer it's RAS: c2_cropped.nii.gz
            # It's then converted to LPS for use here
            im2_file = str(f / "c2_cropped_lps.nii.gz")
        else:
            im2_file = str(f / "c2.nii.gz")

        pt1_file = str(f / "p1.txt")
        pt2_file = str(f / "p2.txt")

        m1 = Mapper(sitk.ReadImage(im1_file))
        m2 = Mapper(sitk.ReadImage(im2_file))

        pt1 = [i[-1] for i in eval(open(pt1_file).read())]
        pt1_ijk = [m1.lps2ijk(*i).tolist() for i in pt1]

        pt2_ijk = find_corresp(model, pt1_ijk, im1_file, im2_file)
        pt2 = [m2.ijk2lps(*i).tolist() for i in pt2_ijk]

        pt2_gt = [i[-1] for i in eval(open(pt2_file).read())]

        df = pd.DataFrame(
            dict(
                idx = f.name,
                pt1 = pt1,
                pt1_ijk = pt1_ijk,
                pt2_pred = pt2,
                pt2_pred_ijk = pt2_ijk,
                pt2_gt = pt2_gt
            )
        )

        df.to_pickle(outfile)

