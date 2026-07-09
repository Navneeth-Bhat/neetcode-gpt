import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # Forward pass
        z1 = np.dot(W1, x) + b1          # (hidden,)  pre-activation
        a1 = np.maximum(0.0, z1)         # (hidden,)  post-ReLU
        z2 = np.dot(W2, a1) + b2         # (out,)     predictions
        loss = np.mean((z2 - y_true)**2)

        # Backward pass
        dz2 = 2 * (z2 - y_true) / len(z2)   # (out,)          image row 1
        dW2 = np.outer(dz2, a1)             # (out, hidden)   image row 2
        db2 = dz2                           # (out,)          image row 3
        da1 = np.dot(W2.T, dz2)             # (hidden,)       image row 4a
        dz1 = da1 * (z1 > 0)                # (hidden,)       image row 4b
        dW1 = np.outer(dz1, x)              # (hidden, in)    image row 5
        db1 = dz1                           # (hidden,)       image row 6

        return {
            'loss': float(np.round(loss, 4)),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist(),
        }