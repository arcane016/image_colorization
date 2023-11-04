import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm
import os
from os.path import join as pjoin
import yaml
import time

from torch.utils.tensorboard import SummaryWriter

from dataloader import ColorizeDataloader
from model import ColorNet
from utils import train_loop, val_loop
import wandb



TRAINING_CONFIG = {
    "loss_fn":"mse_loss",
    "optimizer":"adam",
    "epochs":20,
    "batch_size":128,
    "metrics":"mae"
}

MODEL_CONFIG = {
    "skip_connection": True,
    "downchannels": [1, 64, 128, 256],
    "upchannels": [256, 128, 64, 2],
    "dropout":0.2
}

RUN_INDEX = 5
USE_WANDB = True

if USE_WANDB:
    wandb.login()
    wandb.init(
        project="Image Colorization",
        notes = f"run{RUN_INDEX}, skip={MODEL_CONFIG['skip_connection']}, dropout={MODEL_CONFIG['dropout']}"
    )

EPOCHS = TRAINING_CONFIG["epochs"]
BATCH_SIZE = TRAINING_CONFIG["batch_size"]
DATASET_PATH = "./dataset"
MODEL_LOGS_PATH = f"./logs/run{RUN_INDEX}"
MODEL_CKPT_PATH = pjoin(MODEL_LOGS_PATH, "checkpoints")


if not os.path.exists(MODEL_CKPT_PATH):
    os.makedirs(MODEL_CKPT_PATH)
else:
    if input(f"{MODEL_CKPT_PATH} is okay? if yes, write yes: ") != 'yes':
        raise Exception(f"{MODEL_CKPT_PATH}\nPath already exists. Change RUN_INDEX")

device = ("cuda" if torch.cuda.is_available() else "cpu")
model = ColorNet(
    downchannels=MODEL_CONFIG["downchannels"],
    upchannels=MODEL_CONFIG["upchannels"],
    skip_connect=MODEL_CONFIG["skip_connection"]
).to(device)

train_dataset = ColorizeDataloader(DATASET_PATH, "train")
train_dataloader =  DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, drop_last=True)
val_dataset = ColorizeDataloader(DATASET_PATH, "eval")
val_dataloader =  DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, drop_last=True)

loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters())

writer = SummaryWriter(f"runs/run_{RUN_INDEX}")

with open(pjoin(MODEL_LOGS_PATH, "config.yaml"), 'w') as outfile:
    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    yaml.dump({
        "time": time_string,
        "training_params":TRAINING_CONFIG,
        "model_params": MODEL_CONFIG
    },
    outfile
    )

with open(pjoin(MODEL_LOGS_PATH, "log.txt"), "w") as file:
    file.write("")

def write_in_file(text, file_path=pjoin(MODEL_LOGS_PATH, "log.txt")):
    with open(file_path, "a") as file:
        file.write(text)

for epoch in range(EPOCHS):
    print(f"EPOCH: [{epoch+1}/{EPOCHS}]", end="\n")
    write_in_file(f"EPOCH: [{epoch+1}/{EPOCHS}]\n")

    train_loss, train_mae = train_loop(train_dataloader, model, loss_fn, optim, device)
    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("train_mae", train_mae, epoch)
    write_in_file(f"\tTRAINING  : Loss = {train_loss:>7f}, MAE = {train_mae:>4f}\n")

    val_loss, val_mae = val_loop(val_dataloader, model, loss_fn, device)
    writer.add_scalar("val_loss", val_loss, epoch)
    writer.add_scalar("val_mae", val_mae, epoch)
    write_in_file(f"\tVALIDATION: Loss = {val_loss:>7f}, MAE = {val_mae:>4f}\n\n")

    if USE_WANDB:
        wandb.log({"train_loss":train_loss, "train_mae":train_mae, "val_loss":val_loss, "val_mae":val_mae})

    torch.save(model.state_dict(), pjoin(MODEL_CKPT_PATH, f"ckpt_{epoch+1}.pth"))

writer.flush()

print("\n\nCompleted!!!\n")