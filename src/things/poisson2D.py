# PINN for solving the Poisson equation in 2D: \Delta(u(x,y)) = e^{x*y} * (x^2 + y^2) with Dirichlet boundary conditions. The domain is \omega = [0,1] x [0,1].

import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import qmc
from torch import nn

printData = True

matplotlib.use("qtagg")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


# Feed-Forward Architecture
# MLP architecture
class PoissonPINN(nn.Module):
    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 3,
        activation=nn.Tanh,
    ):
        super().__init__()

        layers = [nn.Linear(in_features, hidden_dim), activation()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation()])
        layers.append(nn.Linear(hidden_dim, out_features))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    # Helps getting stable gradients with Tanh
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


# Problem definition
# Problem domain is [a,b] x [c,d]
a = 0.0
b = 1.0
c = 0.0
d = 1.0


# Boundary condition is e^(x*y)
def g(x, y):
    # return np.zeros_like((x, y))
    return torch.exp(x * y)


def fx(x, y):
    # return np.zeros_like((x, y))
    return torch.exp(x * y) * y


def fy(x, y):
    # return np.zeros_like((x, y))
    return torch.exp(x * y) * x


# Data generation
n_bc = 4
n_data_per_bc = 30

engine = qmc.LatinHypercube(d=1)  # Stratisfied statistical distribution
# Axis 2 represents (x,y, u, ux, uy)
data = np.zeros([n_bc, n_data_per_bc, 3])

for i, j in zip(range(n_bc), [a, b, c, d]):
    points = engine.random(n=n_data_per_bc)[:, 0]

    if i < 2:
        data[i, :, 0] = j
        data[i, :, 1] = points
    else:
        data[i, :, 0] = points
        data[i, :, 1] = j

with torch.no_grad():
    z = torch.as_tensor(data[:, :, :2].reshape(-1, 2), dtype=torch.float64)
    t = g(z[:, 0:1], z[:, 1:2]).cpu().numpy().reshape(n_bc, n_data_per_bc)
    data[:, :, 2] = t

    # For neumann condition
    # tx = fx(z[:, 0:1], z[:, 1:2]).cpu().numpy().reshape(n_bc, n_data_per_bc)
    # ty = fy(z[:, 0:1], z[:, 1:2]).cpu().numpy().reshape(n_bc, n_data_per_bc)
    # data[:, :, 3] = tx
    # data[:, :, 4] = ty

data_flat = data.reshape(n_data_per_bc * n_bc, 3)


# Collocation points
Nc = 500
engine = qmc.LatinHypercube(d=2)
colloc = engine.random(Nc)
colloc = 1 * (colloc - 0)

# torch tensors on the selected device; collocation points require grad for AD
bc_values = torch.as_tensor(data_flat[:, :2], dtype=torch.float64, device=device)
bc_sol_values = torch.as_tensor(data_flat[:, 2:3], dtype=torch.float64, device=device)
collocation_values = torch.as_tensor(colloc, dtype=torch.float64, device=device)

if printData:
    plt.figure("", figsize=(7, 7))
    plt.title("Boundary Data points and Collocation points", fontsize=16)
    plt.scatter(data_flat[:, 0], data_flat[:, 1], marker="x", c="k", label="BDP")
    plt.scatter(colloc[:, 0], colloc[:, 1], s=2, marker=".", c="r", label="CP")
    plt.xlabel("x", fontsize=16)
    plt.ylabel("y", fontsize=16)
    plt.axis("square")
    plt.legend()
    plt.show()


# Build neural network
torch.set_default_dtype(torch.float64)
model = PoissonPINN(2, 1, 32, 3, nn.Tanh).to(device)
print(model)
n_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {n_params}")


def pde_residual(Z):
    Z = Z.detach().requires_grad_(True)
    u0 = model(Z)

    grad = torch.autograd.grad(
        u0, Z, torch.ones_like(u0), create_graph=True, retain_graph=True
    )[0]

    u_x = grad[:, 0:1]
    u_y = grad[:, 1:2]

    u_xx = torch.autograd.grad(
        u_x, Z, torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, Z, torch.ones_like(u_y), create_graph=True)[0][
        :, 1:2
    ]

    return (
        u_xx
        + u_yy
        - (Z[:, 0:1] ** 2 + Z[:, 1:2] ** 2) * torch.exp(Z[:, 0:1] * Z[:, 1:2])
    )


# Training loop
epochs = 15000
lambda_physics = 1.0
lambda_boundary = 20.0

mse_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_values = []

model.train()
start = time.time()

for epoch in range(1, epochs + 1):
    optimizer.zero_grad(set_to_none=True)

    # Compute residual
    residual = pde_residual(collocation_values)
    loss_physics = mse_criterion(residual, torch.zeros_like(residual))

    # Boundary condition evaluation
    loss_bc = mse_criterion(model(bc_values), bc_sol_values)

    total_loss = (lambda_physics * loss_physics) + (lambda_boundary * loss_bc)

    total_loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        loss_values.append(total_loss.item())

    if epoch % 3000 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:5d}/{epochs} | Total Loss: {total_loss.item():.6e} | "
            f"Physics Loss: {loss_physics.item():.6e} | BC Loss: {loss_bc.item():.6e}"
        )


end = time.time()
computation_time = {}
computation_time["pinn"] = end - start
print(f"\ncomputation time: {end - start:.3f} s\n")


if printData:
    plt.figure()
    plt.semilogy(loss_values, label="Total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
