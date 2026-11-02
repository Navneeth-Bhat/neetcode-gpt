import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        n=len(X)
        w=np.zeros(X.shape[1])
        b=0
        for e in range(epochs):
            y_hat = X @ w + b
            mse = (1/n) * sum((y_hat - y)**2)
            dL_dw=(2/n)*X.T@(y_hat-y)
            dL_db=(2/n)*sum(y_hat-y)
            w=w-lr*dL_dw
            b=b-lr*dL_db
        
        return (np.round(w, 5), round(b, 5))

