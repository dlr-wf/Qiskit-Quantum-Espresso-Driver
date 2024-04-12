import numpy as np

# Calculate ERIs via h_ijkl = 4\pi \sum_p (\rho*_il(p) \rho_jk(p))/p²
# with \rho_ij(p)=\int dr \rho_ij(r) e^(-ipr) which is the Fourier transform of
# \rho_ij(r) where \rho_ij(r)=\psi*_i(r)\psi_j(r). Therefore \rho_ij(p)
# is the convolution between \psi*_i(p) and \psi_j(p): \psi*_i(p) * \psi_j(p)
# where \psi_i(p) is the Fourier transform of \psi_i(r)

# The calculation steps are inspired by WEST: https://west-code.org/, https://github.com/west-code-development/West
# Especially the code in the compute_eri_vc function: https://github.com/west-code-development/West/blob/master/Wfreq/solve_eri.f90#L327
# Publication related to WEST:
# Large Scale GW Calculations, M. Govoni and G. Galli, J. Chem. Theory Comput. 11, 2680 (2015)
# GPU Acceleration of Large-Scale Full-Frequency GW Calculations, V. Yu and M. Govoni, J. Chem. Theory Comput. 18, 4690 (2022)


def pair_density_gamma(c_ip_array: np.ndarray) -> np.ndarray:
    r"""Calculate pair density at gamma-point in reciprocal space via Fourier transforms
    Calculates \rho_ij(p)=\psi*_i(p) * \psi_j(p) (* is convolution) which is the
    Fourier transform of \psi*_i(r) * \psi_j(r) (* is standard multiplication)

    The argument c_ip_array is \psi_i(p) for a p-grid in reciprocal space at the gamma point.
    The inverse Fourier transform of \psi_i(p) (or c_ip_array) is \psi_i(r) in real space.
    To calculate \psi*_i(r) * \psi_j(r) (* is standard multiplication) we inverse Fourier
    transform both \psi_i(p) and \psi_j(p) resulting in \psi_i(r) and \psi_j(r)
    and then perform the standard multiplication \psi_i(r) * \psi_j(r) (* is standard multiplication).
    Since \psi_i(r) is real at the gamma-point (does not mean that \psi_i(p) is real) we
    perform a inverse Fourier transform of \psi_i(p) + i \psi_j(p). Then the real and
    imaginary part of the inverse Fourier transform represents \psi_i(r) and \psi_j(r), respectively.
    We then perform a Fourier transform of the calculated real space pair density
    \psi_i(r) * \psi_j(r) (* is standard multiplication) and return the result.

    Args:
        c_ip_array (np.ndarray): Array of coefficients describing the Kohn-Sham orbitals in the plane wave basis, shape (#bands, #grid_size, #grid_size, #grid_size)

    Returns:
        np.ndarray: Pair density in reciprocal space
    """
    # assert np.allclose(c_ip_array, c_ip_array.real), "Wavefunction coefficents should be real but are not! Did you perform a DFT calculation on the gamma-point?"
    assert c_ip_array.ndim == 4
    # assert c_ip_array.shape[1] == c_ip_array.shape[2] == c_ip_array.shape[3]
    # c_ip_shifted = np.array([np.fft.fftshift(x) for x in c_ip])
    c_ip_shifted = np.fft.ifftshift(c_ip_array, axes=(1, 2, 3))
    # c_ip_shifted = c_ip_array
    nbands = c_ip_shifted.shape[0]
    ngrid = c_ip_shifted.shape[1:]
    rho_ij_p = np.zeros((nbands, nbands, *ngrid), dtype=c_ip_shifted.dtype)
    print(f"rho_ij_p.shape: {rho_ij_p.shape}")
    for i in range(nbands):
        for j in range(nbands):
            # Only valid for gamma-point calculation since psi_i(r) is real at the gamma-point
            psi_i_plus_i_times_psi_j = c_ip_shifted[i] + 1.0j * c_ip_shifted[j]
            complex_psi_r = np.fft.ifftn(psi_i_plus_i_times_psi_j)
            # complex_psi_r = np.fft.ifftshift(complex_psi_r)
            rho_ij_r = (
                complex_psi_r.real * complex_psi_r.imag
            )  # only valid for gamma-point
            rho_ij_p_val = np.fft.fftn(
                rho_ij_r
            )  # psi_i(r) * psi_j(r), where * is the standard multiplication
            # Same as psi_i(p) * psi_j(p), where * is the convolution operation
            rho_ij_p[i, j, :] = np.fft.fftshift(rho_ij_p_val)

    return rho_ij_p


def eri_gamma(p: np.ndarray, c_ip: np.ndarray, b: np.ndarray, mill: np.ndarray) -> np.ndarray:
    r"""Calculate Electron Repulsion Integrals (ERIs) via pair densities at the gamma point
    We calculate h_ijkl=4\pi \sum_{p \neq 0} \rho*_il(p)\rho_jk(p)/|p|²
    Since the momenta p and reciprocal space wavefunctions \psi_i(p) (given as c_ip)
    are given as a list, we need to transfer \psi_i(p) to a 3D grid.
    Then \psi_i(p) is represented on a 3D reciprocal space grid.
    We calculate 1/|p|^2 on all grid points resulting in an infinite value at the
    center of the grid. This infinite value is set to zero, therefore,
    technically we later perform a sum over all momenta p except the zero momenta.
    Note that there are different techniques handling with this singularity
    by e.g. using a resolution-of-identity.

    Args:
        p (np.ndarray): Array of momentum vectors, shape (#waves, 3)
        c_ip (np.ndarray): Array of coefficients describing the Kohn-Sham orbitals in the plane wave basis, shape (#bands, #waves)

    Returns:
        np.ndarray: ERIs in reciprocal space
    """
    nbands, nwaves = c_ip.shape  # number of DFT bands and number of plane waves
    
    assert mill.ndim==2 and mill.shape[1]==3, f"Array mill should contain 3D Miller indices but has shape {mill.shape}!"
    assert b.ndim==2 and b.shape[0]==b.shape[1]==3, f"Array b should contain 3 3D reciprocal lattice vectors but has shape {b.shape}!"
    # assert p.shape == mill.shape, f"Array of momentum vectors and Miller indices should have the same shape but have {p.shape} and {mill.shape}, respectively!"

    # # Extract maximum and minimum momenta in each direction
    # p_min_x = p[:, 0].min()
    # p_max_x = p[:, 0].max()
    # p_min_y = p[:, 1].min()
    # p_max_y = p[:, 1].max()
    # p_min_z = p[:, 2].min()
    # p_max_z = p[:, 2].max()
    # # assume equally spaced grid and define spacing as the distance between the first two momenta vectors
    # grid_spacing = np.linalg.norm(p[0] - p[1], ord=2)
    
    # grid_spacing_x = np.linalg.norm(b[0,:], ord=2)
    # grid_spacing_y = np.linalg.norm(b[1,:], ord=2)
    # grid_spacing_z = np.linalg.norm(b[2,:], ord=2)
    
    max_min = mill.max(axis=0) - mill.min(axis=0)

    # Initialize 3D array for each DFT band with zero
    # We therefore set \psi_i(p) on the whole momentum grid for each i
    c_ip_array = np.zeros(
        (
            nbands,
            # Number of grid points for given maximum and minimum momenta and given grid spacing
            *(max_min + 1)
        ),
        dtype=c_ip.dtype,
    )
    # Set \psi_i(p) on given grid points
    for idx, mill_idx in enumerate(mill):
        x, y, z = mill_idx
        i, j, k = (
            x + max_min[0] // 2,
            y + max_min[1] // 2,
            z + max_min[2] // 2,
        )
        c_ip_array[:, i, j, k] = c_ip[:, idx]
    # for idx, coords in enumerate(p):
    #     x, y, z = coords  # momentum vector components in each direction
    #     # index in 3D array belonging to given momentum vector
    #     i, j, k = (
    #         int((x - p_min_x) / grid_spacing),
    #         int((y - p_min_y) / grid_spacing),
    #         int((z - p_min_z) / grid_spacing),
    #     )
    #     # Set \psi_i(p) on given grid point at computed index
    #     c_ip_array[:, i, j, k] = c_ip[:, idx]

    # Calculate |p|² for each grid point
    p_norm_squared_array = np.zeros(c_ip_array.shape[1:])
    for i in range(p_norm_squared_array.shape[0]):
        x = i - max_min[0] // 2
        for j in range(p_norm_squared_array.shape[1]):
            y = j - max_min[1] // 2
            for k in range(p_norm_squared_array.shape[2]):
                z = k - max_min[2] // 2
                p_norm_squared_array[i, j, k] = np.linalg.norm(b[0] * x + b[1] * y + b[2] * z, ord=2)**2

    # for i in range(p_norm_squared_array.shape[0]):
    #     x = (
    #         i * grid_spacing
    #     ) + p_min_x  # Compute momentum vector component for given index
    #     for j in range(p_norm_squared_array.shape[1]):
    #         y = (
    #             j * grid_spacing
    #         ) + p_min_y  # Compute momentum vector component for given index
    #         for k in range(p_norm_squared_array.shape[2]):
    #             z = (
    #                 k * grid_spacing
    #             ) + p_min_z  # Compute momentum vector component for given index
    #             # Calculate norm of momentum vector
    #             p_norm_squared_array[i, j, k] = x**2 + y**2 + z**2
    # Calculate 1/|p|²
    p_norm_squared_array[p_norm_squared_array == 0] = np.inf
    one_over_p_norm_squared_array = 1 / p_norm_squared_array
    # Set infinite values to zero \sum_p -> \sum_{p \neq 0}
    # one_over_p_norm_squared_array[np.isinf(one_over_p_norm_squared_array)] = 0

    # Calculate pair density \rho_ij(p) in reciprocal space
    rho_ij_p = pair_density_gamma(c_ip_array=c_ip_array)
    # Initialize ERI array
    # TODO: We do not need to calculate al matrix elements due to symmetries
    eri = np.zeros((nbands, nbands, nbands, nbands), dtype=c_ip.dtype)

    # Flatten arrays going from the 3D grid to a 1D list for each DFT band
    rho_ij_p = np.reshape(rho_ij_p, newshape=(rho_ij_p.shape[0], rho_ij_p.shape[1], -1))
    one_over_p_norm_squared_array = np.reshape(
        one_over_p_norm_squared_array, newshape=(-1)
    )

    # 4\pi \sum_{p \neq 0} \rho*_il(p)\rho_jk(p)/|p|²
    eri = (
        4
        * np.pi
        * np.einsum(
            "ilp, jkp, p -> ijkl",
            rho_ij_p.conj(),
            rho_ij_p,
            one_over_p_norm_squared_array,
        )
    )

    # Rescaling of Fourier transforms (correct argument?)
    eri *= len(one_over_p_norm_squared_array) ** 2

    return eri
