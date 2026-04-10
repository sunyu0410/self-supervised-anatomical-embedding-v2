import torch
import torch.optim as optim

from model import model
from data import ds, dl

epochs = 50
model = model.train().float()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10, epochs):
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

        print(total_loss)

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'pmcc_train/checkpoints/sam-v1-ft-epoch-{epoch}.pth')

