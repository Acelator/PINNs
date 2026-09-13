import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn

matplotlib.use("qtagg")

# 1. Device configuration & reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# if torch.cuda.is_available():
# print("CUDA SUPPORTED")


# 2. Feed-Forward Architecture (Tanh activation for smooth AD)
class PINN(nn.Module):
    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 3,
    ):
        super().__init__()

        activation = nn.Tanh  # Standard
        # activation = nn.SiLU
        # We can also use Gelu or Sine or Sigmoid (the worst in this case)

        layers = [nn.Linear(in_features, hidden_dim), activation()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation()])
        layers.append(nn.Linear(hidden_dim, out_features))
        self.net = nn.Sequential(*layers)

        # Xavier Normal initialization for stable gradients with Tanh
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


# 3. Problem definition: u'(t) + w0*u(t) = 0, u(0) = 1 over t in [0, 1]
# (Newton Law of cooling)
w0 = 2.0
t_boundary = torch.tensor([[0.0]], dtype=torch.float32, device=device)
u_boundary_target = torch.tensor([[1.0]], dtype=torch.float32, device=device)

# Collocation points
N_colloc = 50
t_colloc = torch.linspace(
    0.0, 1.0, N_colloc, dtype=torch.float32, device=device
).unsqueeze(1)
t_colloc.requires_grad_(True)

# 4. Model, Criterion & Adam Optimizer
model = PINN().to(device)
mse_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 5. Training loop (Adam only, explicit loss weighting)
epochs = 15000  # Number of iterations
lambda_physics = 1.0
lambda_boundary = 20.0

for epoch in range(1, epochs + 1):
    optimizer.zero_grad(set_to_none=True)

    # Boundary condition evaluation
    u_bc_pred = model(t_boundary)
    loss_bc = mse_criterion(u_bc_pred, u_boundary_target)

    # Interior physics residual evaluation
    u_colloc_pred = model(t_colloc)
    dudt = torch.autograd.grad(
        outputs=u_colloc_pred,
        inputs=t_colloc,
        grad_outputs=torch.ones_like(u_colloc_pred),
        create_graph=True,
        retain_graph=True,
    )[0]

    # ODE residual: f(t) = u'(t) + w0 * u(t)
    residual = dudt + w0 * u_colloc_pred
    loss_physics = mse_criterion(residual, torch.zeros_like(residual))

    # Manually scaled composite loss
    total_loss = (lambda_physics * loss_physics) + (lambda_boundary * loss_bc)

    # Backpropagation & update
    total_loss.backward()
    optimizer.step()

    if epoch % 3000 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:5d}/{epochs} | Total Loss: {total_loss.item():.6e} | "
            f"Physics Loss: {loss_physics.item():.6e} | BC Loss: {loss_bc.item():.6e}"
        )

# 6. Evaluation against analytical solution u(t) = exp(-w0 * t)
with torch.no_grad():
    t_test = torch.linspace(
        0.0, 1.0, 200, dtype=torch.float32, device=device
    ).unsqueeze(1)
    u_pred = model(t_test).cpu().numpy()
    t_test_np = t_test.cpu().numpy()
    u_exact = torch.exp(-w0 * t_test).cpu().numpy()

plt.figure(figsize=(8, 4))
plt.plot(t_test_np, u_exact, label=f"Exact: $e^{{-{w0}t}}$", color="black", linewidth=2)
plt.plot(t_test_np, u_pred, "--", label="PINN (Adam)", color="crimson", linewidth=2)
plt.scatter(
    t_colloc.detach().cpu().numpy(),
    [0] * N_colloc,
    marker="|",
    color="gray",
    label="Collocation points",
)
plt.xlabel("t")
plt.ylabel("u(t)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# Suggested improvements by ai:
# Log loss_physics, loss_bc history and compute relative L_2 error vs exact solution
# Then add two-stage optimization: Adam + L-BFGS fine-tune, and switch to float64 for accurate dudt via torch.autograd.grad.
# Only after that, ablate hidden_dim, num_layers, activation in PINN, N_colloc with resampling, and adaptive weighting to replace lambda_boundary=20.0.
