# PINN for solving the Poisson equation in 2D: \Delta(u(x,y)) = e^{x*y} * (x^2 + y^2) with Dirichlet boundary conditions (g as defined below on the boundary). The domain is \omega = [0,1] x [0,1].

import math
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import qmc
from torch import nn

printData = False

# Allowed values are hard and soft. They represent how we should approach the boundary conditions. When hard is set, we impose explicitly the boundary condition.
boundary_type_condition = "hard"

if boundary_type_condition != "hard" and boundary_type_condition != "soft":
    print("Type of boundary constraint not recognised")
    sys.exit("Review boundary_type_condition value")
matplotlib.use("qtagg")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.set_default_dtype(torch.float64)


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

    # Helps obtain stable gradients with Tanh
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if boundary_type_condition == "soft":
            return self.net(t)

        # Hard boundary constraint
        x = t[:, 0:1]
        y = t[:, 1:2]

        nn_out = self.net(t)

        D = x * (1.0 - x) * y * (1.0 - y)

        # Ansatz
        # G defines the boundary condition
        u = G(x, y) + D * nn_out
        return u


# Problem definition
# Problem domain is [a,b] x [c,d]
a = 0.0
b = 1.0
c = 0.0
d = 1.0


# Boundary condition is e^(x*y), but it is also the exact solution. So it shouldn't be used anywhere but on the boundary.
def g(x, y):
    # return np.zeros_like((x, y))
    return torch.exp(x * y)


def G(x, y):
    return 1 - x - y + y * torch.exp(x) + x * torch.exp(y) - x * y * (math.e - 1)


def gx(x, y):
    # return np.zeros_like((x, y))
    return torch.exp(x * y) * y


def gy(x, y):
    # return np.zeros_like((x, y))
    return torch.exp(x * y) * x


# Data generation

# If hard, we don't need boundary points to approximate the solution
if boundary_type_condition == "soft":
    n_bc = 4
    n_data_per_bc = 30

    engine = qmc.LatinHypercube(d=1)  # Stratified statistical distribution
    # Axis 2 represents (x,y, u)
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

        # For Neumann conditions
        # tx = fx(z[:, 0:1], z[:, 1:2]).cpu().numpy().reshape(n_bc, n_data_per_bc)
        # ty = fy(z[:, 0:1], z[:, 1:2]).cpu().numpy().reshape(n_bc, n_data_per_bc)
        # data[:, :, 3] = tx
        # data[:, :, 4] = ty

    data_flat = data.reshape(n_data_per_bc * n_bc, 3)

    bc_values = torch.as_tensor(data_flat[:, :2], dtype=torch.float64, device=device)
    bc_sol_values = torch.as_tensor(
        data_flat[:, 2:3], dtype=torch.float64, device=device
    )


# Collocation points
Nc = 500
engine = qmc.LatinHypercube(d=2)
colloc = engine.random(Nc)
colloc = 1 * (colloc - 0)

# torch tensors on the selected device; collocation points require grad for AD

collocation_values = torch.as_tensor(colloc, dtype=torch.float64, device=device)

if printData:
    plt.figure("", figsize=(7, 7))

    if boundary_type_condition == "soft":
        plt.title("Boundary Data points and Collocation points", fontsize=16)
        plt.scatter(data_flat[:, 0], data_flat[:, 1], marker="x", c="k", label="BDP")
    else:
        plt.title("Collocation points", fontsize=16)
        plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="black", linewidth=1.5)

    plt.scatter(colloc[:, 0], colloc[:, 1], s=2, marker=".", c="r", label="CP")
    plt.xlabel("x", fontsize=16)
    plt.ylabel("y", fontsize=16)
    plt.axis("square")
    plt.legend()
    plt.show()


# Build neural network
model = PoissonPINN(2, 1, 32, 3, nn.Tanh).to(device)
# print(model)
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

    # [:, 0:1] maintaing the dimensionality of the data when extracting
    return (
        u_xx
        + u_yy
        - (Z[:, 0:1] ** 2 + Z[:, 1:2] ** 2) * torch.exp(Z[:, 0:1] * Z[:, 1:2])
    )


# Training loop
epochs = 3000
lambda_physics = 1.0
lambda_boundary = 20.0

mse_criterion = nn.MSELoss()
optimizer_Adam = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_Adam, mode="min", factor=0.5, patience=500
)

loss_values = []


def compute_loss():
    # Compute residual
    residual = pde_residual(collocation_values)
    loss_physics = mse_criterion(residual, torch.zeros_like(residual))

    if boundary_type_condition == "soft":
        # Boundary condition evaluation
        loss_bc = mse_criterion(model(bc_values), bc_sol_values)

        total_loss = (lambda_physics * loss_physics) + (lambda_boundary * loss_bc)

        return total_loss, loss_physics, loss_bc
    else:
        return loss_physics


model.train()
start = time.time()

# for epoch in range(1, epochs + 1):
# print("----- Fase adams -----")
for epoch in range(1, epochs):
    optimizer_Adam.zero_grad(set_to_none=True)

    if boundary_type_condition == "soft":
        loss, loss_physics, loss_bc = compute_loss()
    else:
        loss = compute_loss()

    loss.backward()
    optimizer_Adam.step()
    scheduler.step(loss.detach())

    if epoch % 20 == 0:
        loss_values.append(loss.item())

    if epoch % 3000 == 0 or epoch == 1:
        current_lr = optimizer_Adam.param_groups[0]["lr"]

        if boundary_type_condition == "soft":
            print(
                f"Epoch {epoch:5d}/{epochs} | Total Loss: {loss.item():.6e} | "
                f"Physics Loss: {loss_physics.item():.6e} | BC Loss: {loss_bc.item():.6e} | lr: {current_lr:.6e}"
            )
        else:
            print(
                f"Epoch {epoch:5d}/{epochs} | Total Loss: {loss.item():.6e} | lr: {current_lr:.6e}"
            )

# print("----- Fase LBFGS -----")
# lr=1.0 is normal when using strong_wolfe
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=50000,
    max_eval=50000,
    history_size=50,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    line_search_fn="strong_wolfe",
)

step_counter = 0


def closure():
    global step_counter
    optimizer_lbfgs.zero_grad()

    if boundary_type_condition == "soft":
        loss, loss_physics, loss_bc = compute_loss()
    else:
        loss = compute_loss()

    loss.backward()

    step_counter += 1
    if step_counter % 200 == 0:
        if boundary_type_condition == "soft":
            print(
                f"L-BFGS Eval {step_counter} | Total Loss: {loss.item():.6e} | "
                f"Physics Loss: {loss_physics.item():.6e} | BC Loss: {loss_bc.item():.6e}"
            )
        else:
            print(f"L-BFGS Eval {step_counter} | Total Loss: {loss.item():.6e}")

    return loss


optimizer_lbfgs.step(closure)

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


# Measure error against the real solution (L_2 - norm)

# Dense test points outside the training loop
x_test = torch.linspace(0, 1, 100, device=device, dtype=torch.float64)
y_test = torch.linspace(0, 1, 100, device=device, dtype=torch.float64)
grid_x, grid_y = torch.meshgrid(x_test, y_test, indexing="ij")
pts = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

model.eval()
with torch.no_grad():
    u_pred = model(pts)
    u_true = torch.exp(pts[:, 0] * pts[:, 1]).unsqueeze(1)

    rel_l2_error = torch.norm(u_pred - u_true, p=2) / torch.norm(u_true, p=2)
    print(f"Relative L2 Error: {rel_l2_error.item():.6e}")
