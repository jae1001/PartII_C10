import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

# Define Hückel parameters
beta = -1  # Resonance integral (sets energy scale)

# Function to construct Hamiltonian for given structure
def construct_hamiltonian(meta_substituted=False):
    n_rings = 3 if not meta_substituted else 5
    n_atoms = n_rings * 6  # Each benzene ring has 6 carbon atoms
    H = np.zeros((n_atoms, n_atoms))
    
    # Populate the intra-ring nearest-neighbor connections
    for ring in range(n_rings):
        offset = ring * 6
        for i in range(6):
            H[offset + i, offset + (i + 1) % 6] = beta  # Benzene connectivity
            H[offset + (i + 1) % 6, offset + i] = beta
    
    # Maintain original para connections (C1 to C4)
    H[1, 10] = beta
    H[10, 1] = beta
    
    if meta_substituted:
        # Add meta-substituted connections (C3 and C5 to new rings)
        H[3, 18] = beta  # C3 to new benzene ring
        H[18, 3] = beta
        H[5, 24] = beta  # C5 to another new benzene ring
        H[24, 5] = beta
    
    return H

# Function to compute eigenvalues as function of rotation
def compute_shifted_energies(H, theta_deg):
    theta_rad = np.radians(theta_deg)
    beta_eff = beta * np.cos(theta_rad)  # Adjusted hopping parameter
    
    # Modify Hamiltonian for rotation of C4-linked benzene ring
    H_rotated = H.copy()
    H_rotated[10, 1] = beta_eff  # Adjust para linkage
    H_rotated[1, 10] = beta_eff
    
    # Solve for eigenvalues
    eigenvalues, _ = scipy.linalg.eigh(H_rotated)
    eigenvalues.sort()
    return eigenvalues

# Compute eigenvalues for different rotation angles
angles = np.linspace(0, 90, 10)  # From 0° (coplanar) to 90° (perpendicular)
H_original = construct_hamiltonian(meta_substituted=False)
H_meta_substituted = construct_hamiltonian(meta_substituted=True)

energy_shifts_original = np.array([compute_shifted_energies(H_original, theta) for theta in angles])
energy_shifts_meta = np.array([compute_shifted_energies(H_meta_substituted, theta) for theta in angles])

# Plot energy shifts
fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

for i in range(energy_shifts_original.shape[1]):
    axes[0].plot(angles, energy_shifts_original[:, i], marker='o', linestyle='-', alpha=0.7)
axes[0].set_ylabel("Energy (in units of β)")
axes[0].set_title("Energy Shift: Original 3-Ring Structure")
axes[0].grid(True)

for i in range(energy_shifts_meta.shape[1]):
    axes[1].plot(angles, energy_shifts_meta[:, i], marker='o', linestyle='-', alpha=0.7)
axes[1].set_xlabel("Rotation Angle (degrees)")
axes[1].set_ylabel("Energy (in units of β)")
axes[1].set_title("Energy Shift: Meta-Substituted Structure")
axes[1].grid(True)

plt.tight_layout()
plt.show()
