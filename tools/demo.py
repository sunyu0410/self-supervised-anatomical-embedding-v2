# ssh ysun@vmpr-res-cluster1.unix.petermac.org.au
# [ysun@vmpr-res-cluster1 tools]$ sinteractive -p gpu --mem 64G --gres=gpu:T4 --time=0-12:00
# [ysun@papr-res-gpu02 tools]$ pwd
# /physical_sciences/yusun/self-supervised-anatomical-embedding-v2/tools
# [ysun@papr-res-gpu02 tools]$ cd ..
# [ysun@papr-res-gpu02 self-supervised-anatomical-embedding-v2]$ source py38/bin/activate
# (py38) [ysun@papr-res-gpu02 self-supervised-anatomical-embedding-v2]$ cd tools/
# (py38) [ysun@papr-res-gpu02 tools]$ ipython demo.py 

# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path
from monitor import FileMonitor

sys.path.append('..')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from interfaces import init, get_embedding, get_sim_embed_loc, normalize
from utils import read_image, visualize


os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))  # go to root dir of this project
config_file = 'configs/sam/sam_NIHLN.py'
checkpoint_file = 'checkpoints/SAM.pth'

def get_random_query_point(im):
    im1_shape = list(im['shape'][1:])
    num_try = 0
    bg_th = im['img'].min() + (im['img'].max() - im['img'].min())/10
    while num_try < 20:
        p = np.random.randint(im1_shape)
        if im['img'][p[0], p[1], p[2]] > bg_th:  # not in air
            break
        num_try += 1
    print('query point', p, ', intensity', im['img'][p[0], p[1], p[2]])
    return p


def find_corresp(pt1, im1_file, im2_file):
    # breakpoint()
    time1 = time.time()
    model = init(config_file, checkpoint_file)
    time2 = time.time()
    print('model loading time:', time2 - time1)
    im1, normed_im1, norm_info_1 = read_image(im1_file, is_MRI=False)
    im2, normed_im2, norm_info_2 = read_image(im2_file, is_MRI=False)
    time3 = time.time()
    print('image loading time:', time3 - time2)

    emb1 = get_embedding(normed_im1, model)
    emb2 = get_embedding(normed_im2, model)
    time4 = time.time()
    print('embeddings computing time:', time4 - time3)

    pt1_normed = np.array(pt1) * norm_info_1
    pt2_normed, score = get_sim_embed_loc(emb1, emb2, pt1_normed,
                                          (im2['shape'][3], im2['shape'][1], im2['shape'][2]), norm_info=norm_info_2,
                                          write_sim=False, use_sim_coarse=True)
    pt2 = np.array(pt2_normed).astype(int)
    print(pt2, score)
    time5 = time.time()
    print('matching point computing time:', time5 - time4)
    # breakpoint()
    # visualize(im1['img'], im2['img'], norm_info_1, norm_info_2, pt1, pt2, score, savename='test.png')
    return pt2


if __name__ == '__main__':
    # assume all input image are in torchio "LPS+" direction which equal to "RAI" orientation in ITK-Snap.
    im1_file = 'data/raw_data/NIH_lymph_node/ABD_LYMPH_001.nii.gz'
    im2_file = 'data/raw_data/NIH_lymph_node/ABD_LYMPH_002.nii.gz'

    im1, normed_im1, norm_info_1 = read_image(im1_file, is_MRI=False)
    pt1 = get_random_query_point(im1)
    pt1 = np.array([93, 139,  44])
    pt2 = find_corresp(pt1, im1_file, im2_file)
    
    print(pt1)
    print(pt2)

    file1 = Path('tools/f-1.txt')
    file2 = Path('tools/f-2.txt')
    
    def write_pt2_to_file2():
        pt1 = eval(open(file1).read().strip())
        pt2 = find_corresp(pt1).tolist()
        print(f'Corresponding point: {pt2}')
        output = dict(
            pt1 = pt1,
            pt2 = pt2
        )
        print(str(output), end='', file=open(file2, 'w'))
    
    monitor = FileMonitor(file1, write_pt2_to_file2)
    monitor.start()