#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>

constexpr int ThreadsPerBlock = 512;

// blockDim.x,y,z gives the number of threads in a block, in the particular direction
// gridDim.x,y,z gives the number of blocks in a grid, in the particular direction
// blockDim.x * gridDim.x gives the number of threads in a grid (in the x direction, in this case)

// NVIDIA A100 Information (https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html):
// maximum number of thread blocks per SM is 32

__constant__ float PI = 3.14159265358979323846;

__global__ void sum_p_reduction_ijji(const thrust::complex<double>* c_ip, const double* p_vec, thrust::complex<double>* output,
                                     const int c_ip_size, const int p_size, const int n_bands, const int n_waves) {
    // Calculates h_iijj where i,j are active space indices
    
    // Shapes:
    // c_ip: (#waves * #states), n_waves is fastest changing index
    // p: (#waves * 3), 3 (coordinates) is fastest changing index
    
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    double abs_error = 1e-3;
    
    if (blockId < n_bands * n_bands) {

        int i = blockId % n_bands;
        int j = (blockId / n_bands) % n_bands;

        __shared__ thrust::complex<double> partialSum[ThreadsPerBlock];
        partialSum[threadId] = thrust::complex<double>(0.0f, 0.0f);

        #pragma unroll
        for (int p_index = threadId; p_index < n_waves; p_index += blockDim.x) {
            // c_ip of i-th band of the first wave has index i*n_waves, ... of 2nd wave has index i*n_waves+1, ... of last wave has index (i+1)*n_waves
            double p_x = p_vec[p_index*3+0];
            double p_y = p_vec[p_index*3+1];
            double p_z = p_vec[p_index*3+2];

            thrust::complex<double> c_ip_conj = thrust::conj(c_ip[i*n_waves+p_index]);

            #pragma unroll
            for (int q_index = 0; q_index < n_waves; ++q_index) {
                double q_x = p_vec[q_index*3+0];
                double q_y = p_vec[q_index*3+1];
                double q_z = p_vec[q_index*3+2];

                thrust::complex<double> c_jq_conj = thrust::conj(c_ip[j*n_waves+q_index]);

                #pragma unroll
                for (int s_index = 0; s_index < n_waves; ++s_index) {
                    double s_x = p_vec[s_index*3+0];
                    double s_y = p_vec[s_index*3+1];
                    double s_z = p_vec[s_index*3+2];

                    #pragma unroll
                    for (int r_index = 0; r_index < n_waves; ++r_index) {
                        if (p_index == s_index) {
                            continue;
                        }

                        double r_x = p_vec[r_index*3+0];
                        double r_y = p_vec[r_index*3+1];
                        double r_z = p_vec[r_index*3+2];

                        thrust::complex<double> four_pi_over_p_minus_s_squared = thrust::complex<double>(4.0f * PI /
                                                                                                        ((p_x - s_x)*(p_x - s_x) +
                                                                                                         (p_y - s_y)*(p_y - s_y) +
                                                                                                         (p_z - s_z)*(p_z - s_z)), 0.0f);

                        bool p_minus_r_equals_s_minus_q = (std::abs(p_x - r_x - (s_x - q_x)) <= abs_error &&
                                                           std::abs(p_y - r_y - (s_y - q_y)) <= abs_error &&
                                                           std::abs(p_z - r_z - (s_z - q_z)) <= abs_error);

                        if (p_minus_r_equals_s_minus_q) {
                            partialSum[threadId] += c_ip_conj
                                * c_jq_conj
                                * c_ip[j*n_waves+r_index]
                                * c_ip[i*n_waves+s_index]
                                * four_pi_over_p_minus_s_squared;
                        }
                    }
                }
            }
        }

        // Synchronize the threads within the block to ensure all partial sums are computed
        __syncthreads();

        // Perform reduction to calculate the final sum
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadId < stride) {
                partialSum[threadId] += partialSum[threadId + stride];
            }
            __syncthreads();
        }

        // Store the final sum in global memory
        if (threadId == 0) {
            output[blockId] = partialSum[0];
        }
    }
}

std::tuple<int, int, std::vector<std::complex<double>>> load_coeff(std::string filename) {
    std::ifstream inputFile(filename);
    std::vector<double> coefficients_real;
    std::vector<double> coefficients_imag;

    if (!inputFile)
    {
        std::cerr << "Error opening file " << filename << "!" << std::endl;
        throw std::runtime_error("Exception: Error opening file!");
    }

    int n_bands;
    int n_waves;
    inputFile >> n_bands >> n_waves; // n_waves is fastest changing index

    double real;
    double imag;
    while (inputFile >> real >> imag)
    {
        coefficients_real.push_back(real);
        coefficients_imag.push_back(imag);
    }
    inputFile.close();

    int numCoefficients = coefficients_real.size();
    
    assert((numCoefficients==n_waves*n_bands) && "Number of coefficients does not match number of bands times number of waves written in first line");
    
    std::vector<std::complex<double>> c_ip;
    for (size_t i = 0; i < numCoefficients; ++i) {
        std::complex<double> complexNumber(coefficients_real[i], coefficients_imag[i]);
        c_ip.push_back(complexNumber);
    }

    return {n_bands, n_waves, c_ip};
}

std::vector<double> load_p(std::string filename) {
    std::ifstream inputFile(filename);
    std::vector<double> p_x;
    std::vector<double> p_y;
    std::vector<double> p_z;

    if (!inputFile)
    {
        std::cerr << "Error opening file " << filename << "!" << std::endl;
        throw std::runtime_error("Exception: Error opening file!");
    }

    int n_waves;
    inputFile >> n_waves;

    double x, y, z;
    while (inputFile >> x >> y >> z)
    {
        p_x.push_back(x);
        p_y.push_back(y);
        p_z.push_back(z);
    }
    inputFile.close();

    int numWaves = p_x.size();

    assert((numWaves==n_waves) && "Number of waves written in first line does not match actual number of waves");

    std::vector<double> p;
    for (size_t i = 0; i < numWaves; ++i) {
        p.push_back(p_x[i]);
        p.push_back(p_y[i]);
        p.push_back(p_z[i]);
    }

    return p;
}

std::vector<float> load_occ(std::string filename) {
    std::ifstream inputFile(filename);
    std::vector<float> occs;

    if (!inputFile)
    {
        std::cerr << "Error opening file " << filename << "!" << std::endl;
        throw std::runtime_error("Exception: Error opening file!");
    }

    int n_bands;
    inputFile >> n_bands;

    float occ;
    while (inputFile >> occ)
    {
        occs.push_back(occ);
    }
    inputFile.close();

    int numBands = occs.size();

    assert((numBands==n_bands) && "Number of bands written in first line does not match actual number of bands");

    return occs;
}

bool write_output(thrust::host_vector<thrust::complex<float>> output, std::string filename) {
    // Open the file for writing
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Unable to open the file." << std::endl;
        return false;
    }

    // Write the number of complex elements to the file
    size_t numComplexElements = output.size();
    int n_bands = int(std::pow(numComplexElements, 0.5));
    outFile << numComplexElements << " " << n_bands << "\n";

    // Write each complex number to the file
    outFile << std::setprecision(std::numeric_limits<float>::max_digits10);

    
    for (int idx=0; idx<output.size(); ++idx) {
        int i = idx % n_bands;
        int j = (idx / n_bands) % n_bands;
        outFile << i << " " << j << " " << j << " " << i << " " << output[idx].real() << " " << output[idx].imag() << "\n";
    }

    // Close the file
    outFile.close();
    std::cout << "Data has been written to the file successfully." << std::endl;

    return true;
}

int main(int argc, char* argv[]) {
    std::string base_folder = "../eri/";
    
    if (argc != 3) {
        std::cerr << "Number of arguments needs to be two, start band (int) and end band (int), but is " << argc-1 << "!" << std::endl;
        return 1;
    }
    
    std::cout << "Calculating frozen core mean-field energy component g_ijji" << std::endl;
    // Load coefficients and momentum vectors from file
    std::string filename_coeff {base_folder+"c_ip.txt"};
    std::string filename_p {base_folder+"p.txt"};
    std::string filename_occ {base_folder+"occ_binary.txt"};
    std::tuple<int, int, std::vector<std::complex<double>>> coeff_tuple = load_coeff(filename_coeff);
    int n_bands_all = std::get<0>(coeff_tuple);
    int n_waves_all = std::get<1>(coeff_tuple);
    std::vector<std::complex<double>> c_ip_all = std::get<2>(coeff_tuple);
    std::vector<double> p_all = load_p(filename_p);
    std::vector<float> occ = load_occ(filename_occ);

    std::cout << "occ size: " << occ.size() << std::endl;

    int start_band = std::stoi(argv[1]); // included
    int end_band = std::stoi(argv[2]);   // excluded
    int n_bands = end_band-start_band;
    if (start_band < 0 || start_band >= n_bands_all) {
        std::cerr << "Start band (" << start_band << ") is invalid! Needs to be 0 or larger and must not be equal or larger than "
                  << n_bands_all << std::endl;
        return 1;
    }
    if (end_band < 0 || end_band > n_bands_all || end_band <= start_band) {
        std::cerr << "End band (" << end_band << ") is invalid! Needs to be 0 or larger and must not be equal or larger than "
                  << n_bands_all << " and cannot be equal or smaller than start band (" << start_band << ")!" << std::endl;
        return 1;
    }

    std::cout << "Occupations of bands (selected [" << start_band << ", " << end_band << "] in |...|): ";
    for (int i = 0; i < occ.size(); ++i) {
        if (i == start_band || i == end_band) {
            std::cout << "| ";
        }
        std::cout << occ[i] << " ";
    }
    if (end_band == occ.size()) {
        std::cout << "|";
    }
    std::cout << std::endl;

    int n_waves = n_waves_all;
    std::vector<std::complex<float>> c_ip_bands = std::vector<std::complex<float>>(c_ip_all.begin()+start_band*n_waves_all,
                                                                             c_ip_all.begin()+end_band*n_waves_all);
    std::vector<std::complex<double>> c_ip;
    c_ip.reserve(n_bands*n_waves);
    for (int i=0; i<c_ip_bands.size(); ++i) {
        if ((i%n_waves_all)<n_waves) {
            c_ip.push_back(c_ip_bands[i]);
        }
    }

    std::vector<double> p = std::vector<double>(p_all.begin(),
                                                p_all.begin()+n_waves*3);

    std::cout << "n_bands: " << n_bands << std::endl;
    std::cout << "n_waves: " << n_waves << std::endl;

    size_t size_c_ip = c_ip.size();
    thrust::device_vector<thrust::complex<double>> dev_c_ip = c_ip;
    std::cout << "size_c_ip: " << size_c_ip << std::endl;

    size_t size_p = p.size();
    thrust::device_vector<double> dev_p = p;
    std::cout << "size_p: " << size_p << std::endl;

    int output_size = n_bands*n_bands;
    thrust::device_vector<thrust::complex<double>> devOutput(output_size, thrust::complex<double>(0.0f, 0.0f));

    // Define the block and grid sizes
    int blocks = output_size;
    
    // Launch the kernel
    std::cout << "CUDA kernel launch with " << blocks << " blocks of " << ThreadsPerBlock  << " threads!" << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    sum_p_reduction_ijji<<<blocks, ThreadsPerBlock>>>(thrust::raw_pointer_cast(dev_c_ip.data()), // Takes ~4min for one band
                                                      thrust::raw_pointer_cast(dev_p.data()),
                                                      thrust::raw_pointer_cast(devOutput.data()),
                                                      size_c_ip, size_p, n_bands, n_waves);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n",
            cudaGetErrorString(cudaerr));
        return 1;
    }

    cudaerr = cudaPeekAtLastError();
    if (cudaerr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n",
            cudaGetErrorString(cudaerr));
        return 1;
    }
    cudaEventRecord(stop);
    
    // Copy the result back to the host
    thrust::host_vector<thrust::complex<double>> hostOutput = devOutput;

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed CUDA Time [ms]/[s]/[min]: " << milliseconds << "/" << milliseconds/1000 << "/" << milliseconds/1000/60 << std::endl;

    // Print the result
    std::cout << "output size: " << hostOutput.size() << std::endl;

    std::string filename_output {base_folder+"eri_frozen_ijji_"+std::to_string(start_band)+"_"+std::to_string(end_band)+".txt"};
    bool writeOutput = write_output(hostOutput, filename_output);
    if (writeOutput) {
        std::cout << "Electron repulsion integrals successfully written to " << filename_output << std::endl;
    }
    else {
        std::cout << "Could not write output to file!" << std::endl;
    }

    std::cout << "Finished!" << std::endl;

    return 0;
}
