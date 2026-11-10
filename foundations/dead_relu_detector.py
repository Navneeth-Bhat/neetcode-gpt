import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_fractions=[]
        out=x
        with torch.no_grad():
            for layer in model:
                out=layer(out)
                if isinstance(layer,nn.ReLU):
                    dead=torch.all(out==0,dim=0)
                    dead_fractions.append(round(dead.sum().item()/out.shape[1],4))
        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        if not dead_fractions:
            return []
        max_fraction=None
        strictly_increasing=True
        for i,fraction in enumerate(dead_fractions):
            if fraction>0.5:
                return 'use_leaky_relu'
            if i==0:
                max_fraction=fraction
                if fraction>0.3:
                    return 'reinitialize'
            if i>0 and fraction<dead_fractions[i-1]:
                strictly_increasing=False
            if fraction>max_fraction:
                max_fraction=fraction
        if  strictly_increasing and dead_fractions[-1]>0.1:
            return 'reduce_learning_rate'
        if max_fraction<0.1:
            return 'healthy'
        return 'healthy'
        
