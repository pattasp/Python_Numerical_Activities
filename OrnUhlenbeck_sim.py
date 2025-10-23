# ornstein_uhlenbeck_sim.py
# Simulate (a) non-stationary OU with Y0=0 and (b) stationary OU with Y0 ~ N(0, 1/2)
# SDE: dY_t = -2 Y_t dt + sqrt(2) dW_t

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def simulate_ou_paths(lam=2.0, sigma=np.sqrt(2.0), T=1.0, dt=0.01, n_paths=100,
                      y0_mode="nonstationary", rng=None):
    """
    Exact-discretization simulation of OU:
      Y_{t+dt} = a*Y_t + N(0, noise_var), a = exp(-lam dt),
      noise_var = (sigma^2 / (2*lam)) * (1 - exp(-2*lam dt))

    y0_mode: "nonstationary" -> Y0=0, "stationary" -> Y0~N(0, sigma^2/(2*lam))
    """
    if rng is None:
        rng = np.random.default_rng()

    n_steps = int(T/dt)
    t = np.linspace(0.0, T, n_steps + 1)

    a = np.exp(-lam * dt)
    noise_var = (sigma**2 / (2.0 * lam)) * (1.0 - np.exp(-2.0 * lam * dt))
    noise_std = np.sqrt(noise_var)

    paths = np.zeros((n_paths, n_steps + 1))
    if y0_mode == "stationary":
        # Stationary variance: sigma^2/(2*lam) = 1/2 with our parameters
        paths[:, 0] = rng.normal(0.0, np.sqrt(sigma**2/(2.0*lam)), size=n_paths)
    elif y0_mode == "nonstationary":
        paths[:, 0] = 0.0
    else:
        raise ValueError("y0_mode must be 'nonstationary' or 'stationary'.")

    # Iterate
    for i in range(n_paths):
        for k in range(n_steps):
            paths[i, k+1] = a * paths[i, k] + noise_std * rng.standard_normal()

    return t, paths, a, noise_var

def main():
    # Parameters (match the covariance in the prompt)
    lam = 2.0
    sigma = np.sqrt(2.0)
    T = 1.0
    dt = 0.01
    n_paths = 100
    outdir = Path("ou_outputs")
    outdir.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(12345)  # set seed for reproducibility

    # (a) Non-stationary OU with Y0 = 0
    t, Y_nonstat, a, noise_var = simulate_ou_paths(
        lam=lam, sigma=sigma, T=T, dt=dt, n_paths=n_paths, y0_mode="nonstationary", rng=rng
    )

    # (b) Stationary OU with Y0 ~ N(0, 1/2)
    _, Y_stat, _, _ = simulate_ou_paths(
        lam=lam, sigma=sigma, T=T, dt=dt, n_paths=n_paths, y0_mode="stationary", rng=rng
    )

    # Save arrays
    np.save(outdir / "ou_paths_nonstationary.npy", Y_nonstat)
    np.save(outdir / "ou_paths_stationary.npy", Y_stat)
    np.save(outdir / "time_grid.npy", t)

    # Optional:  save as CSV 
    # np.savetxt(outdir / "time_grid.csv", t, delimiter=",")
    # np.savetxt(outdir / "ou_paths_nonstationary.csv", Y_nonstat, delimiter=",")
    # np.savetxt(outdir / "ou_paths_stationary.csv", Y_stat, delimiter=",")

    # Plot 100 paths (a) Non-stationary
    plt.figure(figsize=(9, 5))
    for i in range(n_paths):
        plt.plot(t, Y_nonstat[i], alpha=0.35)
    plt.title("Ornstein–Uhlenbeck (non-stationary), $Y_0=0$, 100 paths")
    plt.xlabel("t")
    plt.ylabel("$Y_t$")
    plt.tight_layout()
    plt.show()

    # Plot 100 paths (b) Stationary
    plt.figure(figsize=(9, 5))
    for i in range(n_paths):
        plt.plot(t, Y_stat[i], alpha=0.35)
    plt.title("Ornstein–Uhlenbeck (stationary), $Y_0\\sim\\mathcal{N}(0,1/2)$, 100 paths")
    plt.xlabel("t")
    plt.ylabel("$Y_t$")
    plt.tight_layout()
    plt.show()

    # Empirical mean/variance across paths
    emp_mean_nonstat = Y_nonstat.mean(axis=0)
    emp_var_nonstat = Y_nonstat.var(axis=0)

    emp_mean_stat = Y_stat.mean(axis=0)
    emp_var_stat = Y_stat.var(axis=0)

    # Theoretical variances:
    # Var_nonstat(t) = (sigma^2/(2*lam)) * (1 - exp(-2*lam t)) = 0.5 * (1 - e^{-4t})
    theo_var_nonstat = 0.5 * (1.0 - np.exp(-4.0 * t))
    # Var_stat(t) = sigma^2/(2*lam) = 0.5 (constant)
    theo_var_stat = 0.5 * np.ones_like(t)

    # Plot variance comparison: non-stationary
    plt.figure(figsize=(9, 5))
    plt.plot(t, emp_var_nonstat, label="Empirical var (non-stationary)")
    plt.plot(t, theo_var_nonstat, linestyle="--", label="Theoretical var (non-stationary)")
    plt.title("Variance over time: non-stationary OU")
    plt.xlabel("t")
    plt.ylabel("Variance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot variance comparison: stationary
    plt.figure(figsize=(9, 5))
    plt.plot(t, emp_var_stat, label="Empirical var (stationary)")
    plt.plot(t, theo_var_stat, linestyle="--", label="Theoretical var (stationary)")
    plt.title("Variance over time: stationary OU")
    plt.xlabel("t")
    plt.ylabel("Variance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print a quick summary
    print("OU parameters: lambda=2, sigma=sqrt(2)")
    print(f"Discretization: dt={dt}, steps={int(T/dt)}, paths={n_paths}")
    print(f"Exact AR(1) coefficient a = exp(-lambda*dt) = {np.exp(-lam*dt):.6f}")
    print(f"Per-step innovation variance = {(sigma**2/(2*lam))*(1-np.exp(-2*lam*dt)):.6f}")
    print("Files saved under:", outdir.resolve())

if __name__ == "__main__":
    main()
