import torch

def cg(
        A,
        b,
        x0: torch.Tensor = None,
        max_iter: int = None,
        tol: float = 1e-3,
        regularisation = 0.0
):
        ''' Conjugate Gradient method to solve Ax = b. A has to be hermitian and diagonal.'''
        def apply_A(x):
                return A*x + regularisation * x
        j = 0
        r = b - apply_A(x0)
        d = r
        delta = torch.sum(torch.conj(r) * r).real
        delta_0 = delta
        for i in range(max_iter):
                if torch.abs(delta) < torch.abs(tol **2 * delta_0):
                        print(f"Converged at iteration {i}.")
                        break
                q = apply_A(d)
                if torch.sum(torch.conj(d)* q).real < 1e-20:
                        print(f"Breakdown at iteration {i}.")
                        break
                alpha = delta / torch.sum(torch.conj(d)* q).real
                x0 = x0 + alpha * d
                if i % 50 == 0:
                        r = b - apply_A(x0)
                else:
                        r = r - alpha * q
                delta_old = delta
                delta = torch.sum(torch.conj(r) * r).real
                beta = delta / delta_old
                d = r + beta * d
                j += 1
        return x0, j