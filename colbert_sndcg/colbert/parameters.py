import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

SAVED_CHECKPOINTS = [32*1000, 100*1000, 150*1000, 200*1000, 300*1000, 400*1000]
SAVED_CHECKPOINTS += [10*1000, 20*1000, 30*1000, 40*1000, 50*1000, 60*1000, 70*1000, 80*1000, 90*1000]
SAVED_CHECKPOINTS += [25*1000, 50*1000, 75*1000]
SAVED_CHECKPOINTS += [1*1000, 2*1000, 5*1000, 44*1000]
SAVED_CHECKPOINTS += [1]

SAVED_CHECKPOINTS = set(SAVED_CHECKPOINTS)
