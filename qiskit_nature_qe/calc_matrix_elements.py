import numpy as np


def nuclear_repulsion_energy(atoms: list, cell_volume: float) -> float:
    """Calculate nuclear repulsion energy in Hartree units

    Args:
        atoms (list): List of dictionaries describing the involded atoms
        cell_volume (float): Cell volume of the computationen real space cell.

    Returns:
        float: Nuclear repulsion energy
    """
    # nuclear repulsion = 1 / cell_volume \sum_{I<J} Z_I Z_J/|R_I-R_J|
    # Nuclear repulsion energy in Hartree units
    nuclear_repulsion = 0.0
    for i, atom_i in enumerate(atoms):
        pos_i = atom_i["position_hartree"]
        z_i = atom_i["atomic_number"]
        for j, atom_j in enumerate(atoms[i + 1 :]):
            pos_j = atom_j["position_hartree"]
            z_j = atom_j["atomic_number"]
            r = np.linalg.norm(pos_i - pos_j, ord=2)
            # print(r)
            nuclear_repulsion += z_i * z_j / r

    return nuclear_repulsion / cell_volume


def check_symmetry_one_body_matrix(matrix: np.ndarray):
    """Check if given matrix satisfies the symmetries of one-body matrices (hermitian)

    Args:
        matrix (np.ndarray): Matrix to check symmetries of

    Returns:
        bool: Bool specifying of symmetry is satisfied
    """
    matrix_hermitian = matrix.T.conj()
    allclose_hermitian = np.allclose(matrix, matrix_hermitian)
    return allclose_hermitian


def check_symmetry_two_body_matrix(matrix: np.ndarray):
    """Check if given matrix satisfies the symmetries of ERIs

    Args:
        matrix (np.ndarray): Matrix to check symmetries of

    Returns:
        tuple[bool, bool, bool]: Bools specifying if following symmetries are satisfied: swap symmetry, hermitian symmetry, hermitian+swap symmetry
    """
    matrix_swap = matrix.swapaxes(0, 1).swapaxes(2, 3)
    matrix_hermitian = matrix.T.conj()
    matrix_hemitian_swap = matrix.T.conj().swapaxes(0, 1).swapaxes(2, 3)

    allclose_hermitian = np.allclose(matrix, matrix_swap)  # , rtol=1e-5, atol=1e-5)
    allclose_swap = np.allclose(matrix, matrix_hermitian)  # , rtol=1e-5, atol=1e-5)
    allclose_hermitian_swap = np.allclose(
        matrix, matrix_hemitian_swap
    )  # , rtol=1e-5, atol=1e-5)
    return allclose_swap, allclose_hermitian, allclose_hermitian_swap


def iTj(p: np.ndarray, c_ip: np.ndarray) -> np.ndarray:  # Calculates <i|T|j>
    """Calculate kinetic energy matrix in Hartree units in the Kohn-Sham basis

    Args:
        p (np.ndarray): Array of momentum vectors
        c_ip (np.ndarray): Array of coefficients describing the Kohn-Sham orbitals in the plane wave basis

    Returns:
        np.ndarray: Kinetic energy matrix
    """
    # Kinetic energy matrix in Hartree units
    p_norm = np.linalg.norm(p, ord=2, axis=1)
    return 0.5 * c_ip.conjugate() @ np.diag(p_norm) @ c_ip.T


def iUj(p: np.ndarray, c_ip: np.ndarray, atoms: list, cell_volume: float) -> np.ndarray:
    """Calculate nuclear interaction matrix in Hartree units in the Kohn-Sham basis

    Args:
        p (np.ndarray): Array of momentum vectors
        c_ip (np.ndarray): Array of coefficients describing the Kohn-Sham orbitals in the plane wave basis
        atoms (list[dict]):  List of dictionaries describing the involded atoms.
        cell_volume (float): Cell volume of the computationen real space cell.

    Returns:
        np.ndarray: Nuclear interaction matrix
    """
    # Nuclear interaction matrix in Hartree units
    # Calculates <i|U|j>
    # <p|U|q> = 4\pi / cell_volume \sum_I  exp(-i (q-p) . R_I) 1/(q-p)²
    # <i|U|j> = \sum_{p,q,p!=q}  c_{p,i}^* c_{q,j} U_pq = 4\pi / cell_volume \sum_{p,q,p!=q} \sum_I  c_{p,i}^* c_{q,j} exp(-i (q-p) . R_I) 1/(q-p)²
    q = p
    Z_I = []
    R_I = []
    for atom in atoms:
        Z_I.append(atom["atomic_number"])
        R_I.append(atom["position_hartree"])
    Z_I = np.array(Z_I)  # shape (#atoms, 3)
    R_I = np.array(R_I)  # shape (#atoms, 3)

    q_minus_p = q[None] - p[:, None]  # shape (#waves, #waves, 3)
    q_minus_p_norm = np.linalg.norm(
        q_minus_p, ord=2, axis=2
    )  # Has zeros on diagonal, shape (#waves, #waves)
    q_minus_p_norm_filtered = q_minus_p_norm.copy() + np.eye(
        q_minus_p_norm.shape[0]
    )  # Replace zeros on diagonal with ones

    q_minus_p_dot_R = np.sum(
        q_minus_p[None] * R_I[:, None, None], axis=3
    )  # sum over 3D-coordinates, shape (#atoms, #waves, #waves)
    prefactor = (
        4
        * np.pi
        * 1
        / cell_volume
        * np.sum(Z_I[:, None, None] * np.exp(-1j * q_minus_p_dot_R), axis=0)
    )  # sum over atoms, shape (#waves, #waves)
    one_over_q_minus_p_norm_squared = 1 / q_minus_p_norm_filtered**2
    one_over_q_minus_p_norm_squared = one_over_q_minus_p_norm_squared.copy() - np.eye(
        one_over_q_minus_p_norm_squared.shape[0]
    )

    U = prefactor * one_over_q_minus_p_norm_squared
    assert (U.diagonal() == 0.0).all()

    return c_ip.conjugate() @ U @ c_ip.T
