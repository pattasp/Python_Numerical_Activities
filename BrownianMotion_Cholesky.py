#Sim Brownian Motion with Cholesky decomposition 
import numpy as np
import matplotlib.pyplot as plt

#1st step is to fix the step size or discretization. Let it be 0.01=Δt
#Number of total steps is 100
N=100 
dt=0.01
t=np.arange(1,N+1)*dt # t follows the path of 0.01,0.02...until N*dt steps which here is 1.00

#Next step is to define the Cov(B_s,B_t)
#We initialize it as an array of NxN 0-elements equal to steps. Because of Standard Brownian Motion Cov=min(t,s) and E[B_t]=0
C=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        C[i,j]=min(t[i],t[j])
#Cholesky Decomposition
L=np.linalg.cholesky(C)

#Next is a function returning N-IID gaussians
def sample_std_normals(n: int) -> np.ndarray:
    """Return a vector of n i.i.d. N(0,1) random variables."""
    return np.random.normal(0.0, 1.0, size=n)

# Now we generate and plot i.e 100 Brownian motion paths (including first point of B_0 = 0)
np.random.seed(42) 
num_paths = 100

T = np.concatenate([[0.0], t])  # include t=0 at the start where now T time interval 
plt.figure(figsize=(10, 6))

for _ in range(num_paths):
    z = sample_std_normals(N)  # Z ~ N(0, I_N)
    B = L @ z                  # Brownian vector (B_{j/100})_{j=1..100}
    path = np.concatenate([[0.0], B])  # prepend B_0 = 0
    plt.plot(T, path, linewidth=0.8, alpha=0.7)

plt.title("100 Brownian Motion Paths via Cholesky (Δt = 0.01)")
plt.xlabel("t")
plt.ylabel("B(t)")
plt.grid(True, linewidth=0.5, alpha=0.5)
plt.show()