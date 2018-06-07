from tasks.iftask import dataloader
from train import evaluate
import torch

from tasks.iftask import IfTaskModelTraining
model = IfTaskModelTraining()

model.net.load_state_dict(torch.load("./if-task-1000-batch-3000.model"))

_, x, y = next(iter(dataloader(1, 1, 8)))
result = evaluate(model.net, model.criterion, x, y)
print(result['y_out_binarized'])