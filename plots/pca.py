from sklearn.decomposition import PCA
import torch
from skimage.transform import resize

emb1 = torch.load('point_matching_process/emb1.pth')

pca = PCA()

d = emb1[1].numpy()
d2 = d.reshape((128, -1))

# Fit only on the non-air data
mask =  resize(sitk.GetArrayFromImage(ct), d.shape[2:], anti_aliasing=True, preserve_range=True)>0
pca.fit(d2[:, mask.ravel()].T)

# pca.fit(d2.T)

pca.components_
pca.components_.shape

pca.explained_variance_
pca.explained_variance_ratio_

d3 = pca.transform(d2.T)

d3 = d3.T
d4 = d3.reshape((128, ) + d.shape[2:])

d5 = d4[0] # PC1

# Sum the first 10 PC
# d5 = d4[:10].sum(0)

# The first embedding channel
# d5 = d[0,0]

img = sitk.GetImageFromArray(d5)
# img.SetSpacing((4,4,4))
img.SetSpacing([31.57894737, 31.57894737,  7.91666667])
img.SetOrigin(ct.GetOrigin())
img.SetDirection(ct.GetDirection())
sitk.WriteImage(img, 'data/PMCC_ReIrrad_L01/c1_emb0_pc1.nii.gz')

for i in range(128):
    img = sitk.GetImageFromArray(d4[i])
    img.SetSpacing([31.57894737, 31.57894737,  7.91666667])
    img.SetOrigin(ct.GetOrigin())
    img.SetDirection(ct.GetDirection())
    sitk.WriteImage(img, f'data/PMCC_ReIrrad_L01/pca/pc{i}.nii.gz')

import matplotlib.pyplot as plt
plt.imshow(resize(arr[:,250,:], (300, 600)), cmap='gray')
plt.imshow(resize(d4[0,:, 9,:], (300, 600), preserve_range=True), cmap='coolwarm', alpha=0.3)
plt.axis('off')
plt.savefig('data/PMCC_ReIrrad_L01/plot.png')