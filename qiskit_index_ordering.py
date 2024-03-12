import os
from qiskit_nature_qe import qe_driver
from qiskit_nature.second_q.operators import Tensor, PolynomialTensor, FermionicOp
from qiskit_nature.second_q.operators.tensor_ordering import (
    to_physicist_ordering,
    to_chemist_ordering,
    find_index_order,
)

if __name__ == "__main__":
    # Qiskit Index Ordering
    dat_up_file = os.path.join("qe_files", "out_H2", "H2.save", "wfcup1.dat")
    dat_dw_file = os.path.join("qe_files", "out_H2", "H2.save", "wfcdw1.dat")
    xml_file = os.path.join("qe_files", "out_H2", "H2.save", "data-file-schema.xml")

    driver = qe_driver.QE_Driver([dat_up_file, dat_dw_file], xml_file)

    h_ij_up, h_ij_dw = driver.calc_h_ij()
    eri_up, eri_dw, eri_dw_up, eri_up_dw = driver.calc_eri()

    print(
        find_index_order(eri_up)
    )  # pqrs with +_p +_q -_r -_s = +_{{0}} +_{{1}} -_{{2}} -_{{3}}
    print(
        find_index_order(eri_up.swapaxes(1, 2).swapaxes(1, 3))
    )  # pqrs -> psqr with +_p +_q -_r -_s = +_{{0}} +_{{2}} -_{{3}} -_{{1}}
    print(
        find_index_order(to_chemist_ordering(eri_up))
    )  # pqrs -> psqr with +_p +_q -_r -_s = +_{{0}} +_{{2}} -_{{3}} -_{{1}}
    # eri_up.swapaxes(1, 2).swapaxes(1, 3) is equivalent to to_chemist_ordering(eri_up)

    tensor = Tensor(eri_up)  # Given that eri_up is in physicists' order
    tensor.label_template = "+_{{0}} +_{{1}} -_{{2}} -_{{3}}"
    poly = PolynomialTensor({"++--": tensor})
    ferm_op_phys = FermionicOp.from_polynomial_tensor(poly)

    # ferm_op_phys is now identical to the following:
    eri_chem = to_physicist_ordering(eri_up)
    tensor.label_template = "+_{{0}} +_{{2}} -_{{3}} -_{{1}}"
    poly = PolynomialTensor({"++--": eri_chem})
    ferm_op_chem = FermionicOp.from_polynomial_tensor(poly)

    print(ferm_op_chem.equiv(ferm_op_phys))  # True

    # Note:
    # from qiskit_nature.second_q.operators.symmetric_two_body import S1Integrals
    # [S1Integrals](https://qiskit-community.github.io/qiskit-nature/stubs/qiskit_nature.second_q.operators.symmetric_two_body.S1Integrals.html)
    # are ALWAYS in chemists' order
