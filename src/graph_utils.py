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

def compute_chebyshev(adj_mx, K, device):
    adj = torch.FloatTensor(adj_mx).to(device)
    n = adj.shape[0]
    d = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(d + 1e-8, -0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    L = torch.eye(n).to(device) - torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
    lambda_max = torch.linalg.eigvalsh(L).max()
    L_scaled = (2.0 * L / lambda_max) - torch.eye(n).to(device)
    Lk = [torch.eye(n).to(device), L_scaled]
    for k in range(2, K):
        Lk.append(2 * torch.mm(L_scaled, Lk[-1]) - Lk[-2])
    return torch.stack(Lk, dim=0)  # (K, N, N)