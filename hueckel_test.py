import pytest
import numpy as np
from hueckel import construct_hamiltonian, compute_shifted_energies, compute_energy_difference

def test_construct_hamiltonian():
    # Test original 3-ring structure
    H = construct_hamiltonian()
    assert H.shape == (18, 18)
    assert np.sum(H == -1) == 38  # 19 connections * 2 (symmetric matrix)

    # Test meta-substituted structure
    H_meta = construct_hamiltonian(meta_substituted=True)
    assert H_meta.shape == (30, 30)
    assert np.sum(H_meta == -1) == 66  # 33 connections * 2

    # Test para-meta-substituted structure
    H_para_meta = construct_hamiltonian(para_meta_substituted=True)
    assert H_para_meta.shape == (36, 36)
    assert np.sum(H_para_meta == -1) == 80  # 40 connections * 2

def test_compute_shifted_energies():
    H = construct_hamiltonian()
    
    # Test at 0 degrees (no shift)
    energies_0 = compute_shifted_energies(H, 0)
    assert len(energies_0) == 18
    assert np.isclose(np.sum(energies_0), 0, atol=1e-10)  # Sum of eigenvalues should be zero

    # Test at 90 degrees (maximum shift)
    energies_90 = compute_shifted_energies(H, 90)
    assert len(energies_90) == 18
    assert np.isclose(np.sum(energies_90), 0, atol=1e-10)

    # Check that energies are different at 0 and 90 degrees
    assert not np.allclose(energies_0, energies_90)

def test_compute_energy_difference():
    H = construct_hamiltonian()
    energy_diff = compute_energy_difference(H)
    
    # Energy difference should be positive (90° should have higher energy than 0°)
    assert energy_diff > 0

    # Test with meta-substituted structure
    H_meta = construct_hamiltonian(meta_substituted=True)
    energy_diff_meta = compute_energy_difference(H_meta)
    assert energy_diff_meta > 0

    # Test with para-meta-substituted structure
    H_para_meta = construct_hamiltonian(para_meta_substituted=True)
    energy_diff_para_meta = compute_energy_difference(H_para_meta)
    assert energy_diff_para_meta > 0

    # Check that energy differences are different for different structures
    assert not np.isclose(energy_diff, energy_diff_meta)
    assert not np.isclose(energy_diff, energy_diff_para_meta)
    assert not np.isclose(energy_diff_meta, energy_diff_para_meta)