import numpy as np
import torch

def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    # unnormalized laplacian: D - A
    L = np.diag(d) - A
    # normalize by node degrees
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    # scale eigenvalues to [-1, 1]
    lam = np.linalg.eigvals(L).max().real
    return 2 * L / lam - np.eye(n)

def cheb_poly(L, K):
    n = L.shape[0]
    # chebyshev recurrence: T0=I, T1=L, Tk = 2L*T(k-1) - T(k-2)
    LL = [np.eye(n), L[:]]
    for i in range(2, K):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)  # (K, n, n)

def get_laplacian_tensor(adj_mx, K):
    L = scaled_laplacian(adj_mx)
    Lk = cheb_poly(L, K)
    return torch.FloatTensor(Lk.astype(np.float32))