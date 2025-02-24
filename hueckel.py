import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

# Define Hückel parameters
beta = -1  # Resonance integral (sets energy scale)

# Function to construct Hamiltonian for given structure
def construct_hamiltonian(meta_substituted=False,para_meta_substituted=False):
    n_rings = 3
    if meta_substituted: n_rings = 5
    if para_meta_substituted: n_rings = 6

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
        # Add meta-substituted connections (C3 and C6 to new rings)
        H[3, 19] = beta  # C3 to new benzene ring
        H[19, 3] = beta
        H[6, 25] = beta  # C6 to another new benzene ring
        H[25, 6] = beta

    if para_meta_substituted:
        # Add para= meta-substituted connections (C3, C5 and C6 to new rings)
        H[3, 19] = beta  # C3 to new benzene ring
        H[19, 3] = beta
        H[6, 25] = beta  # C6 to another new benzene ring
        H[25, 6] = beta
        H[5, 31] = beta  # C5 to another new benzene ring
        H[31, 8] = beta

    #print(H)
    
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
H_original = construct_hamiltonian(meta_substituted=False, para_meta_substituted=False)
H_meta_substituted = construct_hamiltonian(meta_substituted=True, para_meta_substituted=False)
H_para_meta_substituted = construct_hamiltonian(meta_substituted=False, para_meta_substituted=True)

energy_shifts_original = np.array([compute_shifted_energies(H_original, theta) for theta in angles])
energy_shifts_meta = np.array([compute_shifted_energies(H_meta_substituted, theta) for theta in angles])
energy_shifts_para_meta = np.array([compute_shifted_energies(H_para_meta_substituted, theta) for theta in angles])

# Plot energy shifts
fig, axes = plt.subplots(1, 3, figsize=(8, 10), sharex=True)

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

for i in range(energy_shifts_para_meta.shape[1]):
    axes[2].plot(angles, energy_shifts_para_meta[:, i], marker='o', linestyle='-', alpha=0.7)
axes[2].set_xlabel("Rotation Angle (degrees)")
axes[2].set_ylabel("Energy (in units of β)")
axes[2].set_title("Energy Shift: Para-Meta-Substituted Structure")
axes[2].grid(True)

def compute_energy_difference(H):
    # Compute energies at 0 and 90 degrees
    energies_0 = compute_shifted_energies(H, 0)
    energies_90 = compute_shifted_energies(H, 90)
    
    # Calculate total energy at each angle
    total_energy_0 = np.sum(energies_0[:len(energies_0)//2])  # Assuming half-filling
    total_energy_90 = np.sum(energies_90[:len(energies_90)//2])
    
    # Calculate energy difference
    return total_energy_90 - total_energy_0

# Construct Hamiltonians
H_original = construct_hamiltonian(meta_substituted=False)
H_meta_substituted = construct_hamiltonian(meta_substituted=True)
H_para_meta_substituted = construct_hamiltonian(para_meta_substituted=True)

# Compute energy differences
energy_diff_original = compute_energy_difference(H_original)
energy_diff_meta = compute_energy_difference(H_meta_substituted)
energy_diff_para_meta = compute_energy_difference(H_para_meta_substituted)

# Print results
print(f"Energy difference (90° - 0°) for original 3-ring structure: {energy_diff_original:.4f}β")
print(f"Energy difference (90° - 0°) for meta-substituted structure: {energy_diff_meta:.4f}β")
print(f"Energy difference (90° - 0°) for para-meta-substituted structure: {energy_diff_para_meta:.4f}β")

plt.tight_layout()
plt.show()