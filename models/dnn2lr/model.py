import numpy as np
from typing import List

class DNN2LR:
    def __init__(self):
        self.inconsistency = None
    
    def fit(self, grads: np.array, vals: np.array, emb: List[np.array]):
        """Get cross features based o interpretation inconsistency matrix.

        Args:
            grads: Array with shape [data.shape[0], sum of dims of emb cat cols].
            vals: Array with shape [data.shape[0], len of cat cols].
            emb: List of Array with shape [num of unique vals, dim of emb] for each cat column
            
        """
        emb_dims = [x.shape[1] for x in emb]
        
        csum = [0] + np.cumsum(emb_dims).tolist()
        inconsistency = np.zeros_like(vals, dtype=np.float32)

        for c_ind in range(vals.shape[1]):
            unique = np.unique(vals[:, c_ind])
            
            avg_grad = {}
            for v in unique:
                avg_grad[v] = grads[np.where(vals[:, c_ind] == v)[0], csum[c_ind]:csum[c_ind + 1]].mean(axis=0)
            
            avg_grad_matr = np.vstack([avg_grad[x].reshape(1, -1) for x in vals[:, c_ind]])
            c_vals = np.vstack([emb[c_ind][x].reshape(1, -1) for x in vals[:, c_ind]])
            inconsistency[:, c_ind] = ((grads[:, csum[c_ind]:csum[c_ind + 1]] - avg_grad_matr) * c_vals).sum(axis=1)

        self.inconsistency = inconsistency ** 2
    
    def get_cross_f(self, num: int = None, nu: float = 0.05):
        qnt = np.quantile(self.inconsistency, 1 - nu)
        binary_map = (self.inconsistency >= qnt).astype(int)
        cross_f = binary_map[binary_map.sum(axis=1) > 1]
        hash = [''.join(x) for x in cross_f.astype(str).tolist()]
        vals, indx, cnt = np.unique(hash, return_counts=True, return_index=True)
        sorder = np.argsort(cnt)[::-1]
        ranked_cross_f = [np.where(x != 0)[0].tolist() for x in cross_f[indx[sorder]]]
        return ranked_cross_f[:num if num is not None else len(ranked_cross_f)]
