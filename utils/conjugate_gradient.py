import torch

def my_cg(
        A,
        b,
        x0: torch.Tensor = None,
        max_iter: int = None,
        tol: float = 1e-3,
        regularisation = 0.0
):
        def apply_A(x):
                return A*x + regularisation * x
        j = 0
        r = b - apply_A(x0)
        d = r
        delta = torch.sum(torch.conj(r) * r)
        delta_0 = delta
        for i in range(max_iter):
                if torch.abs(delta) < torch.abs(tol **2 * delta_0):
                        break
                q = apply_A(d)
                alpha = delta / torch.sum(torch.conj(d)* q)
                x0 = x0 + alpha * d
                if i % 10 == 0:
                        print("Iteration:", i, "Residual:", torch.norm(r))
                if i % 50 == 0:
                        r = b - apply_A(x0)
                else:
                        r = r - alpha * q
                delta_old = delta
                delta = torch.sum(torch.conj(r) * r)
                beta = delta / delta_old
                d = r + beta * d
                j += 1
        return x0, j