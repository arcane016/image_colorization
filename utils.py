import torch
import torchvision
from torch import nn

from tqdm import tqdm

def eval_mae(pred, target):
    mae = nn.functional.l1_loss(pred, target)
    return mae

def eval_mse(pred, target):
    mse= nn.functional.mse_loss(pred, target)
    return mse

def train_loop(dataloader, model, loss_fn, optim, device):
    model.train()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0
    total_mae = 0

    pbar = tqdm(dataloader, leave=False)
    pbar.set_description(f"t_loss = 0, t_mae = 0")
    for X0, y0 in pbar:
        X = X0.to(device)
        y = y0.to(device)
        pred = model(X)
        loss = loss_fn(pred, y.type(pred.dtype))
        total_loss += loss
        
        mae = eval_mae(pred, y)
        total_mae += mae

        pbar.set_description(f"t_loss = {loss}, t_mae = {mae}")

        loss.backward()
        optim.step()
        optim.zero_grad()

    train_loss = total_loss/num_batches
    train_mae = total_mae/num_batches

    print(f"\tTRAINING  : Loss = {train_loss:>7f}, MAE = {train_mae:>4f}")

    return train_loss, train_mae

def val_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    val_loss = 0
    val_mae = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, leave=False)
        pbar.set_description(f"v_loss = 0, v_mae = 0")
        for X0, y0 in pbar:
            X = X0.to(device)
            y = y0.to(device)
            pred = model(X)
            loss = loss_fn(pred, y.type(pred.dtype))

            mae = eval_mae(pred, y)

            pbar.set_description(f"v_loss = {loss}, v_mae = {mae}")

            val_loss += loss.item()
            val_mae += mae

    
    val_loss /= num_batches
    val_mae /= num_batches

    print(f"\tVALIDATION: Loss = {val_loss:>7f}, MAE = {val_mae:>4f}")

    return val_loss, val_mae