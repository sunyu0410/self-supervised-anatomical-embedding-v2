import torch
import torch.optim as optim

from model import model
from data import ds, dl

from tqdm import tqdm

epochs = 501
model = model.train().float()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

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

    if epoch % 25 == 0:
        torch.save(model.state_dict(), f'pmcc_train/checkpoints/sam-v1-ft-epoch-{epoch}.pth')

