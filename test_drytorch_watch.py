import torch
from drytorch.lib.objectives import Loss, Metric
from src.data.structures import Targets, Outputs
import sys

def my_ce(outputs, targets): return torch.tensor(1.0)
def my_acc(outputs, targets): return torch.tensor(0.5)

loss = Loss(my_ce, name='CE')
acc = Metric(my_acc, name='Acc')
loss.watch(acc)

print("loss.named_fn:", loss.named_fn)
res = loss.calculate(None, None)
print("calculate:", res)

outputs = Outputs()
targets = Targets(label=torch.tensor(0))
loss.update(outputs, targets)
print("compute:", loss.compute())
