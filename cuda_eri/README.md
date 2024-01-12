# Electron Repulsion Integrals

Calculate electron repulsion integrals from a QuantumEspresso DFT calculation.

**E**lectron **R**epulsion **I**ntegrals are abbreviated with **ERIs** in the following.

## Compilation
Compile with `nvcc` compiler, e.g.: `nvcc eri_sym.cu -o eri_sym.o`

## Usage

We specify the start and end orbitals of the active and core orbitals as parameters passed to the specific executables. Don't giving any arguments shows the needed arguments in the needed order.
- Run [eri_sym.cu](/eri_sym.cu) for calculating $h_{tuvw}$ ERIs and only calculate independent matrix elements
- Run [eri_frozen_core_iijj.cu](/eri_frozen_core_iijj.cu) for calculating $h_{iijj}$ ERIs in the frozen core approximation
- Run [eri_frozen_core_ijji.cu](/eri_frozen_core_ijji.cu) for calculating $h_{ijji}$ ERIs in the frozen core approximation
- Run [eri_frozen_core_tuii.cu](/eri_frozen_core_tuii.cu) for calculating $h_{tuii}$ ERIs in the frozen core approximation
- Run [eri_frozen_core_tiiu.cu](/eri_frozen_core_tiiu.cu) for calculating $h_{tiiu}$ ERIs in the frozen core approximation


Run [eri.cu](/eri.cu) for calculating $h_{tuvw}$ ERIs and calculate **all** matrix elements. Note that we need to set the `start_band` and `end_band` variables in [eri.cu](/eri.cu)

## Features
- Calculate ERIs $h_{tuvw}=\bra{tu}\frac{1}{|\hat{r} _i-\hat{r}_j|}\ket{vw}$ where $\ket{t}$, $\ket{u}$, $\ket{v}$, $\ket{w}$ are Kohn-Sham orbitals expanded in the plane-wave basis. See [formulas](#formulas) section for more information.
- Calculate ERIs for the Frozen Core Approximation (see [this](https://iopscience.iop.org/article/10.1088/2058-9565/abd334/pdf) and [this](https://pubs.aip.org/aip/jcp/article/154/11/114105/315377) paper), namely $h_{iijj}$, $h_{ijji}$, $h_{tuii}$, $h_{tiiu}$ where $i,j$ are core indices and $t,u$ are active indices.


## Implementation Details
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
The calculation of ERIs $h_{tuvw}$ is implemented concurrently using CUDA thread blocks such that each $h_{tuvw}$ is calculated in a thread block. We spawn as many thread blocks as ERIs we want to calculate. Each thread block consists of $512$ threads. The sum over $p$ (see [formulas](#formulas)) is distributed over these $512$ threads.


## Formulas
For the ERIs we calculate 
$$
h_{tuvw}=\bra{tu}V\ket{vw}=\sum_{pqrs}c^\ast_{p,t}c^\ast_{q,u}c_{r,v}c_{s,w}\ \bra{pq}V\ket{rs}=
$$
$$
\sum_{pqrs}c^\ast_{p,t}c^\ast_{q,u}c_{r,v}c_{s,w}\ \frac{4\pi}{|p-s|^2}\delta(p-(r+s-q))\,,
$$
where $p,q,r,s$ are momentum vectors. $c_{p,t}$ are the coefficients defining the Kohn-Sham orbitals $\ket{t}=\sum_G c_{G,t}\ \ket{k+G}$ where $G$ are momentum vectors and $k$ is a momentum vector defining the $k$-point.

## Output Format
The first line in each output file states the number of matrix elements listed and the number of bands.
The rest of the file are the ERIs and their indices, where the first 4 entries of each line correspond to the indices $t,u,v,w$ in $h_{tuvw}$ followed by the real and imaginary part of $h_{tuvw}$.

The specfic information in the header depends on the ERI index type ($tuvw, iijj, ijji, tuii, tiiu$).   
- ERI $h_{tuvw}$: Number of matrix elements listed and number of active bands
- ERIs $h_{iijj}$ and $h_{iijj}$: Number of matrix elements listed and number of core bands
- ERIs $h_{tuii}$ and $h_{tiiu}$: Number of matrix elements listed, number of active bands and number of core bands

## Performance
The runtime scales linearly with the number of ERIs, with the fourth power of the number of plane-waves ( $\mathcal{O}(N^4_\mathrm{pw})$ ) and with the fourth power of the number of bands ( $\mathcal{O}(N^4_\mathrm{bands})$ ). Only calculating independent matrix elements only changes the prefactor in the $\mathcal{O}(N^4_\mathrm{bands})$ scaling, as far as the authors know.

## Notes
#### Floating Point Type
We use $64$-bit floats to calculate the ERIs. We find that using $32$-bit floats leads to $3-5\%$ deviation from the results obtained with $64$-bit floats. Therefore, we encourage the usage of $64$-bit floats although it is computationally more demanding.

## Authors
- [Erik Schultheis](mailto:erik.schultheis@dlr.de)