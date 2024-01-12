from warnings import warn
import numpy as np


class EriHashmap:
    """Hashmap the maps indices to ERIs"""

    def __init__(self, supress_warnings=False) -> None:
        self.hashmap = {}
        self.supress_warnings = supress_warnings

    def update(self, indices: np.ndarray, values: np.ndarray):
        for idx, val in zip(indices, values):
            idx = tuple(idx.tolist())
            if idx in self.hashmap:
                if self.supress_warnings is False:
                    warn(f"Index {idx} already in hashmap")
                continue
            self.hashmap[idx] = val

    def __getitem__(self, key):
        return self.hashmap[key]

    def keys(self):
        return self.hashmap.keys()

    def values(self):
        return self.hashmap.values()

    def items(self):
        return self.hashmap.items()

    def __iter__(self):
        return self.hashmap.__iter__()

    def __str__(self):
        return self.hashmap.__str__()

    def __repr__(self):
        return self.hashmap.__repr__()

    def __len__(self):
        return self.hashmap.__len__()

    def get_tuvw(self, active_orbitals: list[int]):
        n_active = len(active_orbitals)
        if len(active_orbitals) == 0:
            return np.zeros((n_active, n_active, n_active, n_active))
        orbital_min = min(active_orbitals)
        g_tuvw: np.ndarray = np.zeros(
            (n_active, n_active, n_active, n_active), dtype=np.complex128
        )
        for t in active_orbitals:
            for u in active_orbitals:
                for v in active_orbitals:
                    for w in active_orbitals:
                        g_tuvw[
                            t - orbital_min,
                            u - orbital_min,
                            v - orbital_min,
                            w - orbital_min,
                        ] = self[(t, u, v, w)]
        return g_tuvw

    def get_iijj(self, core_orbitals: list[int]):
        if len(core_orbitals) == 0:
            return np.array([[]])
        orbital_min = min(core_orbitals)
        n_core = len(core_orbitals)
        g_iijj: np.ndarray = np.zeros((n_core, n_core), dtype=np.complex128)
        for i in core_orbitals:
            for j in core_orbitals:
                g_iijj[i - orbital_min, j - orbital_min] = self[(i, i, j, j)]
        return g_iijj

    def get_ijji(self, core_orbitals: list[int]):
        if len(core_orbitals) == 0:
            return np.array([[]])
        orbital_min = min(core_orbitals)
        n_core = len(core_orbitals)
        g_ijji: np.ndarray = np.zeros((n_core, n_core), dtype=np.complex128)
        for i in core_orbitals:
            for j in core_orbitals:
                g_ijji[i - orbital_min, j - orbital_min] = self[(i, j, j, i)]
        return g_ijji

    def get_tuii(self, active_orbitals: list[int], core_orbitals: list[int]):
        n_active = len(active_orbitals)
        n_core = len(core_orbitals)
        if len(core_orbitals) == 0 or len(active_orbitals) == 0:
            return np.zeros((n_active, n_active, n_core))
        orbital_min_active = min(active_orbitals)
        orbital_min_core = min(core_orbitals)
        g_tuii: np.ndarray = np.zeros((n_active, n_active, n_core), dtype=np.complex128)
        for t in active_orbitals:
            for i in core_orbitals:
                for u in active_orbitals:
                    g_tuii[
                        t - orbital_min_active,
                        u - orbital_min_active,
                        i - orbital_min_core,
                    ] = self[(t, u, i, i)]
        return g_tuii

    def get_tiiu(self, active_orbitals: list[int], core_orbitals: list[int]):
        n_active = len(active_orbitals)
        n_core = len(core_orbitals)
        if len(core_orbitals) == 0 or len(active_orbitals) == 0:
            return np.zeros((n_active, n_core, n_active))
        orbital_min_active = min(active_orbitals)
        orbital_min_core = min(core_orbitals)
        g_tiiu: np.ndarray = np.zeros((n_active, n_core, n_active), dtype=np.complex128)
        for t in active_orbitals:
            for i in core_orbitals:
                for u in active_orbitals:
                    g_tiiu[
                        t - orbital_min_active,
                        i - orbital_min_core,
                        u - orbital_min_active,
                    ] = self[(t, i, i, u)]
        return g_tiiu


def load_eri_from_file(
    filename: str, idx_type: str
) -> tuple[int, tuple[int, int], np.ndarray, np.ndarray]:
    """_summary_

    Args:
        filename (str): _description_
        idx_type (str): _description_

    Returns:
        tuple[int, tuple[int, int], np.ndarray, np.ndarray]: _description_
    """
    header: np.ndarray = np.loadtxt(
        filename, encoding="utf-8", dtype=np.float64, max_rows=1, skiprows=0
    )
    allowed_header_size = (2, 3)
    assert (
        header.shape[0] in allowed_header_size
    ), f"Expected header with one of {allowed_header_size} entries, but has {header.shape[0]} entries!"
    n_elements = 0
    n_states = (0, 0)
    if header.shape[0] == 2:
        n_elements, n_states = header.astype(int)
        n_states = (0, n_states)
    elif header.shape[0] == 3:
        n_elements, n_states_active, n_states_core = header.astype(int)
        n_states = (n_states_active, n_states_core)
    mat: np.ndarray = np.loadtxt(
        filename, encoding="utf-8", dtype=np.float64, max_rows=None, skiprows=1
    )

    if mat.ndim == 1:  # Add axis if only one matrix elements exist
        mat = mat[None, :]
    indices: np.ndarray = mat[:, :4].astype(int)  # Extract indices as integers

    # Extract matrix elements as complex number
    mat: np.ndarray = mat[:, 4:]
    mat: np.ndarray = np.array([x[0] + x[1] * 1j for x in mat], dtype=np.complex128)

    check_eri_indices(indices, idx_type, n_states)

    return n_elements, n_states, indices, mat


def check_eri_indices(indices: np.ndarray, idx_type: str, n_states: tuple[int, int]):
    energy_idx_types = ("iijj", "ijji")
    potential_idx_types = ("tiiu", "tuii")
    general_idx_type = ("pqrs", "tuvw")
    allowed_idx_types = energy_idx_types + potential_idx_types + general_idx_type
    if idx_type not in allowed_idx_types:
        warn(f"idx_type {idx_type} not in allowed idx_types {allowed_idx_types}!")

    maxs = indices.max(axis=0)
    mins = indices.min(axis=0)

    if idx_type in energy_idx_types:
        assert (
            n_states[0] == 0
        ), f"Expected 0 active states with idx_type {idx_type}, but found {n_states[0]}!"
        assert (
            n_states[1] > 0
        ), f"Expected more than 0 core states with idx_type {idx_type}, but found {n_states[1]}!"
        n_states = n_states[1]
        # Number of indices in a tensor g_iijj or g_ijji where i,j∈{1,..,n_states} should be n_states²
        assert (
            indices.shape[0] == n_states * n_states
        ), f"Expected {n_states*n_states} indices but found {indices.shape[0]}!"

        start_index = mins[0]

        # Check that every possible index combination is in indices
        if idx_type == "iijj":
            for i in range(start_index, start_index + n_states):
                for j in range(start_index, start_index + n_states):
                    assert any(
                        (indices[:] == [i, i, j, j]).all(axis=1)
                    ), f"{[i,i,j,j]} not found in indices!"  # Check if row [i,i,j,j] is present
        elif idx_type == "ijji":
            for i in range(start_index, start_index + n_states):
                for j in range(start_index, start_index + n_states):
                    assert any(
                        (indices[:] == [i, j, j, i]).all(axis=1)
                    ), f"{[i,j,j,i]} not found in indices!"  # Check if row [i,j,j,i] is present
        # Following two asserts should be implicitly checked with the above for loops
        assert (
            maxs[0] == maxs[1] == maxs[2] == maxs[3]
        ), f"Maximum indices are not equal ({maxs[0]}, {maxs[1]}, {maxs[2]}, {maxs[3]})!"
        assert (
            mins[0] == mins[1] == mins[2] == mins[3]
        ), f"Minimum indices are not equal ({mins[0]}, {mins[1]}, {mins[2]}, {mins[3]})!"
    elif idx_type in potential_idx_types:
        assert (
            n_states[0] > 0
        ), f"Expected more than 0 active states with idx_type {idx_type}, but found {n_states[0]}!"
        assert (
            n_states[1] > 0
        ), f"Expected more than 0 core states with idx_type {idx_type}, but found {n_states[1]}!"
        n_states_active: int = n_states[0]
        n_states_core: int = n_states[1]
        # Number of indices in a tensor g_tuii or g_tiiu where i∈{1,..,n_states_core}
        # and t,u∈{1,..,n_states_active} should be n_states_active²*n_states_core
        assert (
            indices.shape[0] == n_states_active * n_states_active * n_states_core
        ), f"Expected {n_states_active*n_states_active*n_states_core} indices but found {indices.shape[0]}!"

        if idx_type == "tiiu":
            start_index_active = mins[0]
            start_index_core = mins[1]
            for t in range(start_index_active, start_index_active + n_states_active):
                for u in range(
                    start_index_active, start_index_active + n_states_active
                ):
                    for i in range(start_index_core, start_index_core + n_states_core):
                        assert any(
                            (indices[:] == [t, i, i, u]).all(axis=1)
                        ), f"{[t,i,i,u]} not found in indices!"  # Check if row [t,i,i,u] is present
            # Following two asserts should be implicitly checked with the above for loop
            assert (
                maxs[0] == maxs[3] and maxs[1] == maxs[2]
            ), f"Maximum indices are not pairwise equal ({maxs[0]}, {maxs[1]}, {maxs[2]}, {maxs[3]})!"
            assert (
                mins[0] == mins[3] and mins[1] == mins[2]
            ), f"Minimum indices are not pairwise equal ({mins[0]}, {mins[1]}, {mins[2]}, {mins[3]})!"
        elif idx_type == "tuii":
            start_index_active = mins[0]
            start_index_core = mins[2]
            for t in range(start_index_active, start_index_active + n_states_active):
                for u in range(
                    start_index_active, start_index_active + n_states_active
                ):
                    for i in range(start_index_core, start_index_core + n_states_core):
                        assert any(
                            (indices[:] == [t, u, i, i]).all(axis=1)
                        ), f"{[t,u,i,i]} not found in indices!"  # Check if row [t,u,i,i] is present
            # Following two asserts should be implicitly checked with the above for loop
            assert (
                maxs[0] == maxs[1] and maxs[2] == maxs[3]
            ), f"Maximum indices are not pairwise equal ({maxs[0]}, {maxs[1]}, {maxs[2]}, {maxs[3]})!"
            assert (
                mins[0] == mins[1] and mins[2] == mins[3]
            ), f"Minimum indices are not pairwise equal ({mins[0]}, {mins[1]}, {mins[2]}, {mins[3]})!"
    elif idx_type in general_idx_type:
        pass  # TODO


if __name__ == "__main__":
    pass
