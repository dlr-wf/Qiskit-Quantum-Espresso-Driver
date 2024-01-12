import os
from wfc import Wfc

if __name__ == "__main__":
    hdf51_file_ncpp = os.path.join("qe_files", "out_H2", "H2.save", "wfc1.hdf5")
    out_xml_ncpp = os.path.join("qe_files", "out_H2", "H2.save", "data-file-schema.xml")

    wfc1_ncpp = Wfc.from_file(hdf51_file_ncpp, out_xml_ncpp)

    overlaps_ncpp = wfc1_ncpp.get_overlaps()

    p = wfc1_ncpp.k_plus_G
    c_ip = wfc1_ncpp.evc
    c_ip_org = wfc1_ncpp.evc_org

    # Save complex array to text file
    c_ip_file = os.path.join("eri", "c_ip.txt")
    with open(c_ip_file, "w", encoding="utf-8") as file:
        file.write(f"{c_ip.shape[0]} {c_ip.shape[1]}\n")
        # file.write("real imag\n")
        for i in range(c_ip.shape[0]):
            for j in range(c_ip.shape[1]):
                file.write(f"{c_ip[i, j].real} {c_ip[i, j].imag}\n")
    print(f"Kohn-Sham coefficients c_i,p written to {c_ip_file}")

    p_file = os.path.join("eri", "p.txt")
    with open(p_file, "w", encoding="utf-8") as file:
        file.write(f"{p.shape[0]}\n")
        # file.write("x y z\n")
        for i in range(p.shape[0]):
            file.write(f"{p[i, 0]} {p[i, 1]} {p[i, 2]}\n")
    print(f"Momentum vector written to {p_file}.")

    occ_file = os.path.join("eri", "occ_binary.txt")
    with open(occ_file, "w", encoding="utf-8") as file:
        file.write(f"{wfc1_ncpp.occupations_binary.shape[0]}\n")
        # file.write("x y z\n")
        for occ in wfc1_ncpp.occupations_binary:
            file.write(f"{occ}\n")
    print(f"Occupation written to {occ_file}.")
