# Electron Repulsion Integrals

Calculate electron repulsion integrals from a QuantumEspresso DFT calculation.

**E**lectron **R**epulsion **I**ntegrals are abbreviated with **ERIs** in the following.

## Features
- Calculate ERIs $h_{tuvw}=\bra{tu}\frac{1}{|\hat{r} _i-\hat{r}_j|}\ket{vw}$ where $\ket{t}$, $\ket{u}$, $\ket{v}$, $\ket{w}$ are Kohn-Sham orbitals expanded in the plane-wave basis. See [formulas](#formulas) section for more information.
- Calculate ERIs for the Frozen Core Approximation (see [this](https://iopscience.iop.org/article/10.1088/2058-9565/abd334/pdf) and [this](https://pubs.aip.org/aip/jcp/article/154/11/114105/315377) paper), namely $h_{iijj}$, $h_{ijji}$, $h_{tuii}$, $h_{tiiu}$ where $i,j$ are core indices and $t,u$ are active indices.

## Formulas
For the ERIs we calculate 
$$
h_{tuvw}=\bra{tu}V\ket{vw}=\sum_{pqrs}c^\ast_{p,t}c^\ast_{q,u}c_{r,v}c_{s,w}\ \bra{pq}V\ket{rs}=
$$
$$
\sum_{pqrs}c^\ast_{p,t}c^\ast_{q,u}c_{r,v}c_{s,w}\ \frac{4\pi}{|p-s|^2}\delta(p-(r+s-q)) = \sum_{qrs}c^\ast_{s-q+r,t}c^\ast_{q,u}c_{r,v}c_{s,w}\ \frac{4\pi}{|r-q|^2}\,,
$$
where $p,q,r,s$ are momentum vectors. $c_{p,t}$ are the coefficients defining the Kohn-Sham orbitals $\ket{t}=\sum_G c_{G,t}\ \ket{k+G}$ where $G$ are momentum vectors and $k$ is a momentum vector defining the $k$-point.

## Implementation Details
#### Obtaining $c_{s-q+r,t}$
The momentum vectors $p$ are stored as a 2-dimensional array $A_p$ of shape $N_\mathrm{pw}\times3$ where $N_\mathrm{pw}$ are the number of plane-waves. The coefficients $c_{p,t}$ are stored as 2-dimensional array $A_{c_{p,t}}$  of shape $N_\mathrm{bands}\times N_\mathrm{pw}$ where $N_\mathrm{bands}$ is the number of Kohn-Sham orbitals used in the calculation of the ERIs. Additionally, we store the Miller indices corresponding to the momentum vectors in a 2-dimensional array $A_m$  of shape $N_\mathrm{pw}\times3$. To get the coefficient $c_{s-q+r,t}$ we use a hashmap which maps Miller indices to coefficients. For this we store the Miller indices that correspond to the momentum vectors $s-q+r$ in a 4-dimensional array where the first 3 dimensions correspond to the indices of $s,q,r$ in $A_p$ and the last index represents the cartesian axis coordinates $x,y,z$ of the Miller indices. We then find the Miller indices of $s-q+r$ in the hashmap to get the coefficients $c_{s-q+r,t}$.   
**Ideas to improve finding/calculating $c_{s-q+r,t}$ are greatly appreaciated!**

#### Independent ERIs
Not all $h_{tuvw}$ are independent. $h_{tuvw}$ obey the following symmetries:
$$
\begin{align*}
h_{tuvw}&=h_{utwv}\quad\mathrm{(Hermiticity)}\\
h_{tuvw}&=h^\ast_{wvut}\quad\mathrm{(Swap Symmetry)}\\
h_{tuvw}&=h^\ast_{vwtu}\quad\mathrm{(Hermiticity+Swap)}\,.
\end{align*}
$$
We use these symmetries to only calculate independent matrix elements when calculating $h_{tuvw}$. Nevertheless, the output file contains all ERIs not only the independent ERIs.

#### Concurrency
The calculation of ERIs $h_{tuvw}$ is implemented concurrently using [rayon](https://github.com/rayon-rs/rayon) such that each $h_{tuvw}$ can be calculated in a separate thread.

## Requirements
- The [Rust programming language](https://www.rust-lang.org/). See [here](https://www.rust-lang.org/learn/get-started) and [here](https://forge.rust-lang.org/infra/other-installation-methods.html) for installation instructions.
- QuantumEspresso with HDF5 support

## Usage
1. Run a QuantumEspresso SCF DFT calculation with HDF5 support.
2. Store the `data-file-schema.xml` and the `wfcX.hdf5` (`X` corresponds to the $k$-point index in the QuantumEspresso calculation) file as `wfc1.hdf5` in the `input` folder.
3. Compile with `cargo build` (unoptimized debug build) or `cargo build --release` (optimized release build). The executable is created in `target/debug` or `target/release`. Compile and run with `cargo run` or `cargo build --release`.
4. Run executable `eri_rs --help` or `cargo run --release -- --help` for information about the arguments, e.g.   
`cargo run --release -- -i tuvw --start-core 0 --end-core 22 --start-active 22 --end-active 26 --n-threads 160`   
to calculate the $h_{tuvw}$ ERIs for the indices $t,u,v,w\in\{22,\ldots,25\}$ with $160$ rayon threads (here the given core indices have no effect since $t,u,v,w$ are all active indices).
5. The calculated ERIs are written in the `output` folder.

## Output Format
The first line in each output file states the number of matrix elements listed and the number of bands.
The rest of the file are the ERIs and their indices, where the first 4 entries of each line correspond to the indices $t,u,v,w$ in $h_{tuvw}$ followed by the real and imaginary part of $h_{tuvw}$.

The specfic information in the header depends on the ERI index type ($tuvw, iijj, ijji, tuii, tiiu$).   
- ERI $h_{tuvw}$: Number of matrix elements listed and number of active bands
- ERIs $h_{iijj}$ and $h_{iijj}$: Number of matrix elements listed and number of core bands
- ERIs $h_{tuii}$ and $h_{tiiu}$: Number of matrix elements listed, number of active bands and number of core bands

## Performance
The runtime scales linearly with the number of ERIs, cubically with the number of plane-waves ( $\mathcal{O}(N^3_\mathrm{pw})$ ) and with the fourth power of the number of bands ( $\mathcal{O}(N^4_\mathrm{bands})$ ). Only calculating independent matrix elements only changes the prefactor in the $\mathcal{O}(N^4_\mathrm{bands})$ scaling, as far as the authors know.

We found that the calculation of **one** ERI takes about $150$ seconds on an Intel Xeon Gold 6240R CPU @ 2.40GHz. Considering as many threads as ERIs and perfect parallelism the time to calculate all ERIs is constant should be equivalent to the time to calculate one ERI.

## Notes
#### Floating Point Type
We use $32$- or $64$-bit floats to calculate the ERIs (defined by the `EriType` type in [main.rs](src/main.rs)). We find that using $32$-bit floats leads to $3-5\%$ deviation from the results obtained with $64$-bit floats. Therefore, we encourage the usage of $64$-bit floats although it is computationally more demanding.

#### Large RAM Usage
We are aware of the large RAM usage ($\sim30$ GB) which is mainly because of storing all Miller indices and coefficients in a hashmap that maps Miller indices to coefficients, although the hashmap only stores references to the coefficients.

## Authors
- [Erik Schultheis](mailto:erik.schultheis@dlr.de)
