# Fine tuned using lr=1e-5 for 50 epochs
# Now keep fine-tuning using lr=1e-6 for 50 epochs

import torch
import torch.optim as optim

from model import model
from data import ds, dl

from tqdm import tqdm
from mmcv.runner import load_checkpoint

epochs = 15
checkpoint = load_checkpoint(model, 'pmcc_train/checkpoints_lr1e-6/sam-v1-ft-epoch-10.pth', map_location="cpu")
model = model.train().float()
optimizer = optim.Adam(model.parameters(), lr=1e-6)

pbar = tqdm(range(epochs))
for epoch in pbar:
    for img, meshgrid, valid, metas in dl:

        optimizer.zero_grad()

        # --- FORWARD PASS ---
        # We call forward_train directly to get the loss dictionary
        losses = model.forward_train(
            img=img,
            img_metas=metas,
            meshgrid=meshgrid,
            valid=valid,
        )

        # --- LOSS CALCULATION ---
        # losses is a dict: {'loss_1': tensor, 'loss_2': tensor}
        # We must sum them into one scalar for .backward()
        total_loss = sum(_loss.mean() for _loss in losses.values())

        # --- BACKWARD & OPTIMIZE ---
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix_str(f'Loss: {total_loss.item()}')

    if epoch % 1 == 0:
        torch.save(model.state_dict(), f'pmcc_train/checkpoints_lr1e-7/sam-v1-ft-epoch-{epoch}.pth')

