use hdf5::File;
use ndarray as nd;
use ndarray::{array, s, Array1, Array2, Axis};
use ndarray_linalg::{Norm, Scalar};
use num_complex::Complex;
use std::collections::HashMap;
use std::fs::File as stdFile;
use std::io::Write;
use std::time::Instant;
use std::{f64::consts::PI, path::PathBuf};
extern crate xml;
use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use std::fs::File as File_std;
use std::io::BufReader;
use xmltree::Element;
use std::sync::atomic::{AtomicUsize, Ordering};

type EriType = f64; // Error in eri matrix elements due to f32 is about 3%-5%.

struct Wfc {
    evc: Array2<Complex<EriType>>,
    mill: Array2<i32>,
    k_plus_g: Array2<EriType>,
    occupations_binary: Array1<bool>,
}

#[derive(Debug)]
enum WfcError {
    FileOpen(String),
    GammaOnly(String),
    Attribute(String),
    Dataset(String),
    Xml(String),
}

impl std::fmt::Display for WfcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WfcError::FileOpen(msg) => write!(f, "WfcError::FileOpen: {}", msg),
            WfcError::GammaOnly(msg) => write!(f, "WfcError::GammaOnly: {}", msg),
            WfcError::Attribute(msg) => write!(f, "WfcError::Attribute: {}", msg),
            WfcError::Dataset(msg) => write!(f, "WfcError::Dataset: {}", msg),
            WfcError::Xml(msg) => write!(f, "WfcError::Xml: {}", msg),
        }
    }
}

fn read_wfc_hdf5(path: &PathBuf, path_xml: &PathBuf) -> Result<Wfc, WfcError> {
    let file = File::open(path)
        .map_err(|e| WfcError::FileOpen(format!("Could not open file {}. {e}", path.display())))?; // open for reading

    // let _igwx: i32 = file.attr("igwx").unwrap().read_scalar().unwrap();
    // let _ik: i32 = file.attr("ik").unwrap().read_scalar().unwrap();
    // let _ispin: i32 = file.attr("ispin").unwrap().read_scalar().unwrap();
    // let _nbnd: i32 = file.attr("nbnd").unwrap().read_scalar().unwrap();
    // let _ngw: i32 = file.attr("ngw").unwrap().read_scalar().unwrap();
    // let _npol: i32 = file.attr("npol").unwrap().read_scalar().unwrap();
    // let _scale_factor: f32 = file.attr("scale_factor").unwrap().read_scalar().unwrap();
    
    let gamma_only_str = file
        .attr("gamma_only")
        .map_err(|e| WfcError::Attribute(format!("Could not read attribute 'gamma_only': {e}")))?
        .read_scalar::<hdf5::types::FixedAscii<25>>()
        .map_err(|e| WfcError::Attribute(format!("Could not transform attribute 'gamma_only' into scalar: {e}")))?
        .as_str()
        .to_string();
    let gamma_only = match gamma_only_str {
        x if x == ".TRUE." => Ok(true),
        x if x == ".FALSE." => Ok(false),
        _ => Err(WfcError::GammaOnly(format!(
            "'gamma_only' ({gamma_only_str}) attribute cannot be parsed into bool!"
        ))),
    }?;

    let evc: Array2<EriType> = file.dataset("evc")
        .map_err(|e| WfcError::Dataset(format!("Could not read dataset 'evc': {e}")))?
        .read()
        .map_err(|e| WfcError::Dataset(format!("Could not read dataset 'evc': {e}")))?;
    let evc_real: Array2<Complex<EriType>> = evc.slice(s![..,0..;2]).map(|x| Complex::new(*x, 0.0));
    let evc_imag: Array2<Complex<EriType>> = evc.slice(s![..,1..;2]).map(|x| Complex::new(0.0, *x));
    let mut evc_complex: Array2<Complex<EriType>> = evc_real + evc_imag;
    let mut mill: Array2<i32> = file.dataset("MillerIndices")
        .map_err(|e| WfcError::Dataset(format!("Could not read dataset 'MillerIndices': {e}")))?
        .read()
        .map_err(|e| WfcError::Dataset(format!("Could not read dataset 'MillerIndices': {e}")))?;
    
    let b1_vec: Vec<[EriType; 3]> = file
        .dataset("MillerIndices")
        .map_err(|e| WfcError::Dataset(format!("Could not read dataset 'MillerIndices': {e}")))?
        .attr("bg1")
        .map_err(|e| WfcError::Attribute(format!("Could not read attribute 'MillerIndices.bg1': {e}")))?
        .read_raw::<[EriType; 3]>()
        .map_err(|e| WfcError::Attribute(format!("Could not transform attribute 'MillerIndices.bg1' into vec: {e}")))?;
    let b1: Array1<EriType> = b1_vec.iter().flat_map(|&x| Vec::from(x)).collect();
    let b2_vec: Vec<[EriType; 3]> = file
        .dataset("MillerIndices")
        .map_err(|e| WfcError::Dataset(format!("Could not read dataset 'MillerIndices': {e}")))?
        .attr("bg2")
        .map_err(|e| WfcError::Attribute(format!("Could not read attribute 'MillerIndices.bg2': {e}")))?
        .read_raw::<[EriType; 3]>()
        .map_err(|e| WfcError::Attribute(format!("Could not transform attribute 'MillerIndices.bg2' into vec: {e}")))?;
    let b2: Array1<EriType> = b2_vec.iter().flat_map(|&x| Vec::from(x)).collect();
    let b3_vec: Vec<[EriType; 3]> = file
        .dataset("MillerIndices")
        .map_err(|e| WfcError::Dataset(format!("Could not read dataset 'MillerIndices': {e}")))?
        .attr("bg3")
        .map_err(|e| WfcError::Attribute(format!("Could not read attribute 'MillerIndices.bg3': {e}")))?
        .read_raw::<[EriType; 3]>()
        .map_err(|e| WfcError::Attribute(format!("Could not transform attribute 'MillerIndices.bg3' into vec: {e}")))?;
    let b3: Array1<EriType> = b3_vec.iter().flat_map(|&x| Vec::from(x)).collect();

    let b: Array2<EriType> = nd::stack![Axis(0), b1, b2, b3];

    let mut g: Array2<EriType> = mill.map(|&x| x as EriType).dot(&b);

    if gamma_only {
        let evc_conj: Array2<Complex<EriType>> = evc_complex.map(|x| x.conj());
        evc_complex
            .append(Axis(1), evc_conj.slice(s![.., 1..]).view())
            .map_err(|e| WfcError::GammaOnly(format!("Could not construct 'evc_complex' from 'evc_conj' slice: {e}")))?;
        g.append(Axis(0), g.clone().map(|&x| -x).slice(s![1.., ..]).view())
            .map_err(|e| WfcError::GammaOnly(format!("Could not construct 'g' from 'g' slice: {e}")))?;
        mill.append(Axis(0), mill.clone().map(|&x| -x).slice(s![1.., ..]).view())
            .map_err(|e| WfcError::GammaOnly(format!("Could not construct 'mill' from 'mill' slice: {e}")))?;
    }
    let k: Array1<EriType> = array![0.0, 0.0, 0.0];

    let k_plus_g: Array2<EriType> = g.clone() + k;

    let xml_file = File_std::open(path_xml)
        .map_err(|e| WfcError::FileOpen(format!("Could not open file {}. {e}", path_xml.display())))?;
    let xml_file = BufReader::new(xml_file);
    let names_element = Element::parse(xml_file)
        .map_err(|e| WfcError::FileOpen(format!("Could not open file {:?}. {e}", path_xml.display())))?;
    let _name = names_element.get_child("input"); // atomic_structure, atomic_positions
    let ks_energies = names_element
        .get_child("output")
        .ok_or_else(|| WfcError::Xml(format!("Could not read child 'output' from xml file {:?}", path_xml.display())))?
        .get_child("band_structure")
        .ok_or_else(|| WfcError::Xml(format!("Could not read child 'band_structure' from xml file {:?}", path_xml.display())))?
        .get_child("ks_energies")
        .ok_or_else(|| WfcError::Xml(format!("Could not read child 'ks_energies' from xml file {:?}", path_xml.display())))?;
    let occupations_child = ks_energies.get_child("occupations")
        .ok_or_else(|| WfcError::Xml(format!("Could not read child 'occupations' from xml file {:?}", path_xml.display())))?;

    let occupations: Array1<f32> = occupations_child
        .get_text()
        .ok_or_else(|| WfcError::Xml(format!("Could not transform 'occupations' into text from xml file {:?}", path_xml.display())))?
        .to_string()
        .split_whitespace()
        .map(|x| x.parse::<f32>())
        .collect::<Result<Array1<f32>, _>>()
        .map_err(|e| WfcError::Xml(format!("Could not parse 'occupations' into floats: {e}")))?;
    let occupations_binary: Array1<bool> = occupations.map(|x| (x - 1.0).abs() < x.abs());

    Ok(Wfc {
        evc: evc_complex,
        mill,
        k_plus_g,
        occupations_binary,
    })
}

fn get_independent_indices(n: &usize) -> Vec<Vec<usize>> {
    let mut mat: Vec<Vec<Vec<Vec<bool>>>> = vec![vec![vec![vec![false; *n]; *n]; *n]; *n];
    let mut independent_indices: Vec<Vec<usize>> = Vec::new();
    for l in 0..*n {
        for k in 0..*n {
            for j in 0..*n {
                for i in 0..*n {
                    if mat[i][j][k][l] {
                        continue;
                    }

                    mat[i][j][k][l] = true;
                    mat[j][i][l][k] = true; // V_ijkl = V _jilk (swap)
                    mat[l][k][j][i] = true; // V_ijkl = V*_lkji (hermiticity)
                    mat[k][l][i][j] = true; // V_ijkl = V*_klij (swap+hermiticity)

                    independent_indices.push(vec![i, j, k, l])
                }
            }
        }
    }

    independent_indices
}

#[derive(Debug)]
enum CalculationError {
    Error(String),
}

impl std::fmt::Display for CalculationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalculationError::Error(msg) => write!(f, "CalculationError::Error: {}", msg),
        }
    }
}

fn e_e_interaction_qrs_par(
    p: &Array2<EriType>,
    c_ip: &Array2<Complex<EriType>>,
    mill: &Array2<i32>,
    indices: &Vec<Vec<usize>>,
) -> Result<Vec<Complex<EriType>>, CalculationError> {
    // Laptop: -s, Workstation 488: -s
    // Shapes:
    // p: (#waves, 3)
    // c_ip: (#states, #waves)
    // mill: (#waves, #3)
    let p_complex: nd::Array2<Complex<EriType>> = p.map(|&x| Complex::new(x, 0.0));

    let mut mill_to_c: HashMap<nd::ArrayView1<i32>, nd::ArrayView1<Complex<EriType>>> =
        HashMap::new();
    for (i, val) in mill.axis_iter(nd::Axis(0)).enumerate() {
        mill_to_c.insert(val, c_ip.slice(s![.., i]));
    }

    let p_max = p
        .map_axis(nd::Axis(1), |x| x.norm_l2())
        .into_iter()
        .reduce(EriType::max)
        .ok_or_else(|| CalculationError::Error("Could not find maximum norm of momentum vectors.".to_string()))?;
    let p_max_squared = p_max.powi(2);

    println!("Building hashmap mapping Miller indices to coefficients...");
    let now = Instant::now();
    // When run memory usage increases as the array gets filled.
    // Can we allocate the memory for s_minus_q_plus_r_mill beforehand?
    // nd::Array4::<i32>::zeros and a for loop with indexing does not work memory-wise.
    let s_minus_q_plus_r_mill = nd::Array4::<i32>::from_shape_fn(
        (
            mill.len_of(Axis(0)),
            mill.len_of(Axis(0)),
            mill.len_of(Axis(0)),
            mill.len_of(Axis(1)),
        ),
        |(q, r, s, x)| {
            if (q % 10 == 0) && (r % mill.len_of(Axis(0)) == 0) && (s % mill.len_of(Axis(0)) == 0) && (x == 0) {
                print!("{:?}/{:?}", q, mill.len_of(Axis(0)));
                std::io::stdout().flush().unwrap_or_default();
                print!("\r");
            }
            // Unwrap cannot fail, because q,r,s,x run over mill indices we are accessing. Is there a better way?
            mill.get((s, x)).unwrap_or_else(|| panic!("Unexpected index out of bounds for s={s}, x={x} with lenghts {} and {}", mill.len_of(Axis(0)), mill.len_of(Axis(1)))) -
            mill.get((q, x)).unwrap_or_else(|| panic!("Unexpected index out of bounds for q={q}, x={x} with lenghts {} and {}", mill.len_of(Axis(0)), mill.len_of(Axis(1)))) +
            mill.get((r, x)).unwrap_or_else(|| panic!("Unexpected index out of bounds for r={r}, x={x} with lenghts {} and {}", mill.len_of(Axis(0)), mill.len_of(Axis(1))))
        },
    );
    println!("s_minus_q_plus_r_mill Elapsed: {:.2?}", now.elapsed());
    println!(
        "s_minus_q_plus_r_mill shape: {:#?}",
        s_minus_q_plus_r_mill.shape()
    );

    let c_ip_conj = c_ip.map(|&x| x.conj());

    let p_max_squared_tol = 1e-3;

    let num_indices = indices.len();
    let num_processed: AtomicUsize = AtomicUsize::new(0);

    println!("Calculating ERIs...");
    Ok(indices
        .par_iter()
        .map(|idx| {
            let i = idx[0];
            let j = idx[1];
            let k = idx[2];
            let l = idx[3];

            // Calculate c*_{s-q+r} c*_{q} c_{r} c_{s} 1/|s-q|²

            let c_jq_conj: nd::ArrayView1<Complex<EriType>> = c_ip_conj.slice(s![j, ..]);
            let c_kr: nd::ArrayView1<Complex<EriType>> = c_ip.slice(s![k, ..]);
            let c_ls: nd::ArrayView1<Complex<EriType>> = c_ip.slice(s![l, ..]);

            let mut res: Complex<EriType> = Complex::new(0.0, 0.0);
            for (q_index, q) in p_complex.axis_iter(nd::Axis(0)).enumerate() {
                for (r_index, r) in p_complex.axis_iter(nd::Axis(0)).enumerate() {
                    if q_index == r_index {
                        // momentum vectors are unique and only equal if indices are equal
                        continue;
                    }

                    let q_minus_r = &q - &r;
                    let pot_val = (PI as EriType) * 4.0 / q_minus_r.dot(&q_minus_r);
                    for (s_index, s) in p_complex.axis_iter(nd::Axis(0)).enumerate() {
                        let s_minus_q_plus_r = -&q_minus_r + s;
                        if s_minus_q_plus_r.dot(&s_minus_q_plus_r).re()
                            > p_max_squared + p_max_squared_tol
                        {
                            continue;
                        }

                        // let c_qrs = mill_to_c
                        //     .get(&s_minus_q_plus_r_mill.slice(s![q_index, r_index, s_index, ..]));

                        // // If s-q+r Miller index is not in mill_to_c it is mapped to zero. Unwraps for c_jq_conj, c_kr, c_ls cannot panic
                        // if let Some(val) = c_qrs {
                        //     if let Some(val_i) = val.get(i) {
                        //         res += val_i.conj()
                        //             * c_jq_conj.get(q_index).unwrap_or(&Complex::new(0.0, 0.0))
                        //             * c_kr.get(r_index).unwrap_or(&Complex::new(0.0, 0.0))
                        //             * c_ls.get(s_index).unwrap_or(&Complex::new(0.0, 0.0))
                        //             * pot_val;
                        //     }
                        // }

                        let c_iqrs = mill_to_c
                            .get(&s_minus_q_plus_r_mill.slice(s![q_index, r_index, s_index, ..]))
                            .unwrap()
                            .get(i)
                            .unwrap();

                        res += c_iqrs.conj()
                            * c_jq_conj.get(q_index).unwrap()
                            * c_kr.get(r_index).unwrap()
                            * c_ls.get(s_index).unwrap()
                            * pot_val;
                    }
                }
            }
            let prev: usize = num_processed.fetch_add(1, Ordering::SeqCst);
            println!("Processed {:?}/{:?}", prev+1, num_indices);
            res
        })
        .collect())
}

fn write_eri_to_file_w_indices(
    filename: &str,
    eri: &[Complex<EriType>],
    indices: &[Vec<usize>],
    header: &str,
) -> std::io::Result<()> {
    // Open the file for writing
    let mut file = stdFile::create(filename)?;

    writeln!(file, "{}", header)?;

    // Write each complex number to the file
    for (idx, complex_number) in indices.iter().zip(eri.iter()) {
        let i = idx[0];
        let j = idx[1];
        let k = idx[2];
        let l = idx[3];

        // Extract the real and imaginary parts of the complex number
        let (real, imag) = (complex_number.re, complex_number.im);

        // Convert the real and imaginary parts to strings
        let real_str = real.to_string();
        let imag_str = imag.to_string();

        let idx_str = format!("{i} {j} {k} {l}");

        // Write the line to the file
        let line = format!("{idx_str} {real_str} {imag_str}");
        writeln!(file, "{}", line)?;
    }

    println!("Data written to {} successfully!", filename);

    Ok(())
}

#[derive(Debug, Clone)]
enum IdxType {
    tuvw,
    iijj,
    ijji,
    tuii,
    tiiu,
}

impl std::fmt::Display for IdxType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl ValueEnum for IdxType {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            IdxType::tuvw,
            IdxType::iijj,
            IdxType::ijji,
            IdxType::tuii,
            IdxType::tiiu,
        ]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        Some(self.to_string().into())
    }
}

// Triple / are diplayed as descriptions and help messages when executing the program
/// Calculation of electron repulsion integrals in the basis of Kohn-Sham orbitals which are expanded in plane-waves
#[derive(Parser, Debug)]
#[command(author("Erik Schultheis, erik.schultheis@dlr.de"), version("0.1"), about, long_about = None)]
#[command(
    help_template = "{about-section}Version: {version} Author: {author}\n{usage-heading} {usage} \n{all-args} {tab}"
)]
struct Args {
    /// ERI index type
    #[arg(value_enum, short, long)]
    idx_type: IdxType,

    /// Core start band (included)
    #[arg(short('c'), long("start-core"))]
    start_band_core: usize,

    /// Core end band (excluded)
    #[arg(short('d'), long("end-core"))]
    end_band_core: usize,

    /// Active space start band (included)
    #[arg(short('a'), long("start-active"))]
    start_band_active: usize,

    /// Active space end band (excluded)
    #[arg(short('b'), long("end-active"))]
    end_band_active: usize,

    /// Number rayon threads
    #[arg(short('n'), long("n-threads"), default_value_t = 1)]
    n_threads: usize,
}

fn calculate_tuvw(
    k_plus_g: &Array2<EriType>,
    evc: &Array2<Complex<EriType>>,
    mill: &Array2<i32>,
    indices: &Vec<Vec<usize>>,
    start_band: &usize,
    end_band: &usize,
    save_folder: &[&str],
) -> Result<(), String> {
    let n_bands = end_band - start_band;

    let now = Instant::now();
    let v_ijkl_active = e_e_interaction_qrs_par(k_plus_g, evc, mill, indices)
        .map_err(|e| format!("{e}"))?;
    println!("ijVkl Elapsed: {:.2?}", now.elapsed());

    let mut v_ijkl_full: Vec<Complex<EriType>> =
        vec![Complex::<EriType>::new(0.0, 0.0); n_bands * n_bands * n_bands * n_bands];

    for (idx, elem) in indices.iter().zip(v_ijkl_active.iter()) {
        let i = idx[0] - start_band;
        let j = idx[1] - start_band;
        let k = idx[2] - start_band;
        let l = idx[3] - start_band;

        let ijkl = i + n_bands * j + n_bands * n_bands * k + n_bands * n_bands * n_bands * l;
        let jilk = j + n_bands * i + n_bands * n_bands * l + n_bands * n_bands * n_bands * k;
        let lkji = l + n_bands * k + n_bands * n_bands * j + n_bands * n_bands * n_bands * i;
        let klij = k + n_bands * l + n_bands * n_bands * i + n_bands * n_bands * n_bands * j;

        v_ijkl_full[ijkl] = *elem; // Independent element
        v_ijkl_full[jilk] = *elem; // Swap symmetry
        v_ijkl_full[lkji] = *elem; // Hermiticity symmetry
        v_ijkl_full[klij] = *elem; // Swap+hermiticity symmetry
    }

    let mut indices_full: Vec<Vec<usize>> =
        Vec::with_capacity(n_bands * n_bands * n_bands * n_bands);
    for (idx, _) in v_ijkl_full.iter().enumerate() {
        let i = idx % n_bands + start_band;
        let j = (idx / n_bands) % n_bands + start_band;
        let k = (idx / (n_bands * n_bands)) % n_bands + start_band;
        let l = (idx / (n_bands * n_bands * n_bands)) % n_bands + start_band;

        indices_full.push(vec![i, j, k, l]);
    }

    let eri_type_str = std::any::type_name::<EriType>();
    let filename_tmp = format!("eri_sym_rs_tuvw_{start_band}_{end_band}_{eri_type_str}.txt");
    let filename = filename_tmp.as_str();
    let filename_path_tmp = save_folder
        .iter()
        .chain([filename].iter())
        .collect::<PathBuf>();
    let filename_path = filename_path_tmp.to_str()
        .ok_or_else(|| format!("Could not open file {:?}", filename_path_tmp.display()))?;
    let num_complex_elements = v_ijkl_full.len();
    let header: String = format!("{num_complex_elements} {n_bands}");
    write_eri_to_file_w_indices(filename_path, &v_ijkl_full, &indices_full, &header)
        .map_err(|e| format!("Could not write ERIs to file: {e}"))?;

    Ok(())
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let idx_type = args.idx_type;
    let start_band_core = args.start_band_core;
    let end_band_core = args.end_band_core;
    let start_band_active = args.start_band_active;
    let end_band_active = args.end_band_active;
    let n_threads = args.n_threads;

    println!(
        "Using index type {}\nStart/end band core {}/{}\nStart/end band active {}/{}\nUsing {} rayon threads",
        idx_type, start_band_core, end_band_core, start_band_active, end_band_active, n_threads
    );

    let eri_type_str = std::any::type_name::<EriType>();
    println!("Using type {} to represent eri elements!", eri_type_str);

    // Calculate number of bands
    let n_bands_core = end_band_core - start_band_core;
    let n_bands_active = end_band_active - start_band_active;

    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .map_err(|e| format!("Failed to create rayon thread pool: {e}"))?;

    let input_folder = ["..", "qe_files", "out_H2", "H2.save"];
    let output_folder = ["..", "eri"];

    let path: PathBuf = input_folder.iter().chain(["wfc1.hdf5"].iter()).collect();
    println!(
        "Kohn-Sham orbital information loaded from {}",
        path.display()
    );

    let path_xml: PathBuf = input_folder
        .iter()
        .chain(["data-file-schema.xml"].iter())
        .collect();
    println!("Data file schema loaded from {}", path_xml.display());

    let wfc1: Wfc = read_wfc_hdf5(&path, &path_xml).map_err(|e| format!("Failed reading wfc hdf5 file: {e}"))?;

    let n_bands = wfc1.evc.shape().first()
        .ok_or_else(|| format!("Could not read number of bands from first dimension from 'evc' with shape {:?}", wfc1.evc.shape()))?;
    if start_band_active >= *n_bands {
        return Err(format!("Active space start band ({}) is invalid! Needs to be 0 or larger and must not be equal or larger than number of bands ({})", start_band_active, n_bands));
    }
    if end_band_active > *n_bands || end_band_active <= start_band_active {
        return Err(format!("Active space end band ({}) is invalid! Needs to be 0 or larger and must not be equal or larger than number of bands ({}) and cannot be equal or smaller than active space start band {}", end_band_active, n_bands, start_band_active));
    }
    if start_band_core >= *n_bands {
        return Err(format!("Core space start band ({}) is invalid! Needs to be 0 or larger and must not be equal or larger than number of bands ({})", start_band_core, n_bands));
    }
    if end_band_core > *n_bands || end_band_core <= start_band_core {
        return Err(format!("Core space end band ({}) is invalid! Needs to be 0 or larger and must not be equal or larger than number of bands ({}) and cannot be equal or smaller than active space start band {}", end_band_core, n_bands, start_band_active));
    }

    let mut indep_idx_active = get_independent_indices(&n_bands_active);
    for idx_vec in indep_idx_active.iter_mut() {
        for idx in idx_vec.iter_mut() {
            *idx += start_band_active;
        }
    }

    println!("Occupations of bands (selected in |...|): !");
    for (i, &occ) in wfc1.occupations_binary.iter().enumerate() {
        match idx_type {
            IdxType::tuvw => {
                if i == start_band_active || i == end_band_active {
                    print!("| ");
                }
            }
            IdxType::ijji | IdxType::iijj => {
                if i == start_band_core || i == end_band_core {
                    print!("| ");
                }
            }
            IdxType::tuii | IdxType::tiiu => {
                if i == start_band_active || i == end_band_active {
                    print!("| ");
                }
                if i == start_band_core || i == end_band_core {
                    print!("| ");
                }
            }
        }
        print!("{} ", occ as usize);
    }
    match idx_type {
        IdxType::tuvw => {
            if end_band_active == wfc1.occupations_binary.len() {
                print!("| ");
            }
        }
        IdxType::ijji | IdxType::iijj => {
            if end_band_core == wfc1.occupations_binary.len() {
                print!("| ");
            }
        }
        IdxType::tuii | IdxType::tiiu => {
            if end_band_active == wfc1.occupations_binary.len()
                || end_band_core == wfc1.occupations_binary.len()
            {
                print!("| ");
            }
        }
    }
    println!();

    match idx_type {
        IdxType::tuvw => {
            println!(
                "Calculating {} matrix elements with {} threads!",
                indep_idx_active.len(),
                rayon::current_num_threads()
            );
            calculate_tuvw(
                &wfc1.k_plus_g,
                &wfc1.evc,
                &wfc1.mill,
                &indep_idx_active,
                &start_band_active,
                &end_band_active,
                &output_folder,
            )?
        }
        // Frozen Core Approximation
        IdxType::iijj => {
            let mut iijj_indices: Vec<Vec<usize>> = Vec::with_capacity(n_bands_core * n_bands_core);
            for i in start_band_core..end_band_core {
                for j in start_band_core..end_band_core {
                    iijj_indices.push(vec![i, i, j, j]); // Add every i,i,j,j to indices
                }
            }
            assert!(iijj_indices.len() == n_bands_core * n_bands_core);
            println!(
                "Calculating {} matrix elements with {} threads!",
                iijj_indices.len(),
                rayon::current_num_threads()
            );

            let now = Instant::now();
            let v_iijj =
                e_e_interaction_qrs_par(&wfc1.k_plus_g, &wfc1.evc, &wfc1.mill, &iijj_indices)
                    .map_err(|e| format!("{e}"))?;
            println!("ijVkl Elapsed: {:.2?}", now.elapsed());

            let eri_type_str = std::any::type_name::<EriType>();
            let filename_tmp =
                format!("eri_sym_rs_iijj_{start_band_core}_{end_band_core}_{eri_type_str}.txt");
            let filename = filename_tmp.as_str();
            let filename_path_tmp = output_folder
                .iter()
                .chain([filename].iter())
                .collect::<PathBuf>();
            let filename_path = filename_path_tmp.to_str()
                .ok_or_else(|| format!("Could not convert filename {:?} to str", filename_path_tmp.display()))?;
            let num_complex_elements = v_iijj.len();
            let header: String = format!("{num_complex_elements} {n_bands_core}");
            write_eri_to_file_w_indices(filename_path, &v_iijj, &iijj_indices, &header).
            map_err(|e| format!("Could not write ERIs to file: {e}"))?;
        }
        IdxType::ijji => {
            let mut ijji_indices: Vec<Vec<usize>> = Vec::with_capacity(n_bands_core * n_bands_core);
            for i in start_band_core..end_band_core {
                for j in start_band_core..end_band_core {
                    ijji_indices.push(vec![i, j, j, i]); // Add every i,j,j,i to indices
                }
            }
            assert!(ijji_indices.len() == n_bands_core * n_bands_core);
            println!(
                "Calculating {} matrix elements with {} threads!",
                ijji_indices.len(),
                rayon::current_num_threads()
            );

            let now = Instant::now();
            let v_ijji =
                e_e_interaction_qrs_par(&wfc1.k_plus_g, &wfc1.evc, &wfc1.mill, &ijji_indices)
                    .map_err(|e| format!("{e}"))?;
            println!("ijVkl Elapsed: {:.2?}", now.elapsed());

            let eri_type_str = std::any::type_name::<EriType>();
            let filename_tmp =
                format!("eri_sym_rs_ijji_{start_band_core}_{end_band_core}_{eri_type_str}.txt");
            let filename = filename_tmp.as_str();
            let filename_path_tmp = output_folder
                .iter()
                .chain([filename].iter())
                .collect::<PathBuf>();
            let filename_path = filename_path_tmp.to_str()
                .ok_or_else(|| format!("Could not convert filename {:?} to str", filename_path_tmp.display()))?;
            let num_complex_elements = v_ijji.len();
            let header: String = format!("{num_complex_elements} {n_bands_core}");
            write_eri_to_file_w_indices(filename_path, &v_ijji, &ijji_indices, &header)
                .map_err(|e| format!("Could not write ERIs to file: {e}"))?;
        }
        IdxType::tuii => {
            let mut tuii_indices: Vec<Vec<usize>> =
                Vec::with_capacity(n_bands_active * n_bands_active * n_bands_core);
            for t in start_band_active..end_band_active {
                for u in start_band_active..end_band_active {
                    for i in start_band_core..end_band_core {
                        tuii_indices.push(vec![t, u, i, i]); // Add every t,u,i,i to indices
                    }
                }
            }
            assert!(tuii_indices.len() == n_bands_active * n_bands_active * n_bands_core);
            println!(
                "Calculating {} matrix elements with {} threads!",
                tuii_indices.len(),
                rayon::current_num_threads()
            );

            let now = Instant::now();
            let v_tuii =
                e_e_interaction_qrs_par(&wfc1.k_plus_g, &wfc1.evc, &wfc1.mill, &tuii_indices)
                    .map_err(|e| format!("{e}"))?;
            println!("ijVkl Elapsed: {:.2?}", now.elapsed());

            let eri_type_str = std::any::type_name::<EriType>();
            let filename_tmp = format!("eri_sym_rs_tuii_active_{start_band_active}_{end_band_active}_core_{start_band_core}_{end_band_core}_{eri_type_str}.txt");
            let filename = filename_tmp.as_str();
            let filename_path_tmp = output_folder
                .iter()
                .chain([filename].iter())
                .collect::<PathBuf>();
            let filename_path = filename_path_tmp.to_str()
                .ok_or_else(|| format!("Could not convert filename {:?} to str", filename_path_tmp.display()))?;
            let num_complex_elements = v_tuii.len();
            let header: String = format!("{num_complex_elements} {n_bands_active} {n_bands_core}");
            write_eri_to_file_w_indices(filename_path, &v_tuii, &tuii_indices, &header)
                .map_err(|e| format!("Could not write ERIs to file: {e}"))?;
        }
        IdxType::tiiu => {
            let mut tiiu_indices: Vec<Vec<usize>> =
                Vec::with_capacity(n_bands_active * n_bands_core * n_bands_active);
            for t in start_band_active..end_band_active {
                for i in start_band_core..end_band_core {
                    for u in start_band_active..end_band_active {
                        tiiu_indices.push(vec![t, i, i, u]); // Add every t,i,i,u to indices
                    }
                }
            }
            assert!(tiiu_indices.len() == n_bands_active * n_bands_core * n_bands_active);
            println!(
                "Calculating {} matrix elements with {} threads!",
                tiiu_indices.len(),
                rayon::current_num_threads()
            );

            let now = Instant::now();
            let v_tiiu =
                e_e_interaction_qrs_par(&wfc1.k_plus_g, &wfc1.evc, &wfc1.mill, &tiiu_indices)
                    .map_err(|e| format!("{e}"))?;
            println!("ijVkl Elapsed: {:.2?}", now.elapsed());

            let eri_type_str = std::any::type_name::<EriType>();
            let filename_tmp = format!("eri_sym_rs_tiiu_active_{start_band_active}_{end_band_active}_core_{start_band_core}_{end_band_core}_{eri_type_str}.txt");
            let filename = filename_tmp.as_str();
            let filename_path_tmp = output_folder
                .iter()
                .chain([filename].iter())
                .collect::<PathBuf>();
            let filename_path = filename_path_tmp.to_str()
                .ok_or_else(|| format!("Could not convert filename {:?} to str", filename_path_tmp.display()))?;
            let num_complex_elements = v_tiiu.len();
            let header: String = format!("{num_complex_elements} {n_bands_active} {n_bands_core}");
            write_eri_to_file_w_indices(filename_path, &v_tiiu, &tiiu_indices, &header)
                .map_err(|e| format!("Could not write ERIs to file: {e}"))?;
        }
    };

    Ok(())
}
