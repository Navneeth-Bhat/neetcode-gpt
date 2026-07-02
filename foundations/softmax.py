import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        maxi=max(z)
        result=np.exp(z-maxi)/np.sum(np.exp(z-maxi))
        return np.round(result, 4)
        
