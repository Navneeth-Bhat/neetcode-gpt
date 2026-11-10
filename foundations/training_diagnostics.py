import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        out=x
        stats=[]
        with torch.no_grad():
            for layer in model:
                out=layer(out)
                if isinstance(layer,nn.Linear):
                    mean=round(out.mean().item(),4)
                    std=round(out.std().item(),4)
                    dead= torch.all(out<=0,dim=0)
                    dead_fraction=round(dead.sum().item()/out.shape[1],4)
                    stats.append({"mean":mean,"std":std,"dead_fraction":dead_fraction})

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        pred=model(x)
        loss=nn.MSELoss()(pred,y)
        loss.backward()
        stats=[]
        for layer in model:
            if isinstance(layer,nn.Linear):
                grad=layer.weight.grad
                mean=round(grad.mean().item(),4)
                std=round(grad.std().item(),4)
                norm=round(torch.norm(grad).item(),4)
                stats.append({"mean":mean,"std":std,"norm":norm})
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for i in range(len(activation_stats)):
            if activation_stats[i]['dead_fraction']>0.5:
                return 'dead_neurons'
            if gradient_stats[i]['norm']>1000:
                    return 'exploding_gradients'
            if gradient_stats[i]['norm']<1e-5:
                return 'vanishing_gradients'
            if activation_stats[i]['std']<0.1:
                return 'vanishing_gradients'
            if activation_stats[i]['std']>10.0:
                return 'exploding_gradients'        
        return 'healthy'
