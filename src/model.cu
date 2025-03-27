#include <cuda_runtime.h>
#include <iostream>
#include <float.h>
#include <vector>

static const int blockSize = 256;

__global__ void updatePsiPosVelKernel(double* psi, double* psiVel, double* psiForces, double m, double dt, int numPositions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPositions) {
        psi[idx] += psiVel[idx] * dt + dt * dt * psiForces[idx] / 2.0 / m;
        psiVel[idx] += psiForces[idx] * dt / m;
    }
}

__global__ void updatePsiLengthsKernel(double* psi, double* psiLengths, double a, int numPositions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPositions - 1) {
        double diff = psi[idx + 1] - psi[idx];
        psiLengths[idx] = sqrt(diff * diff + a * a);
    }
}

__global__ void updatePsiForcesKernel(double* psi, double* psiVel, double* psiLengths, double* psiForces, double k1, double g1, double alpha, int numPositions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < numPositions - 1) {
        psiForces[idx] = k1 * (1.0 - alpha / psiLengths[idx]) * (psi[idx + 1] - psi[idx]);
        psiForces[idx] += g1 * (psiVel[idx + 1] - psiVel[idx]);
        psiForces[idx] -= k1 * (1.0 - alpha / psiLengths[idx - 1]) * (psi[idx] - psi[idx - 1]);
        psiForces[idx] -= g1 * (psiVel[idx] - psiVel[idx - 1]);
    }
}

extern "C" void updatePsiPosVelCUDA(double* psi, double* psiVel, double* psiForces, double m, double dt, int numPositions) {
    int gridSize = (numPositions + blockSize - 1) / blockSize;
    updatePsiPosVelKernel<<<gridSize, blockSize>>>(psi, psiVel, psiForces, m, dt, numPositions);
    cudaDeviceSynchronize();
}

extern "C" void updatePsiForcesCUDA(double* psi, double* psiVel, double* psiLengths, double* psiForces, double k1, double g1, double a, double alpha, int numPositions) {
    int gridSize = (numPositions + blockSize - 1) / blockSize;
    updatePsiLengthsKernel<<<gridSize, blockSize>>>(psi, psiLengths, a, numPositions);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    updatePsiForcesKernel<<<gridSize, blockSize>>>(psi, psiVel, psiLengths, psiForces, k1, g1, alpha, numPositions);
    cudaDeviceSynchronize();
}

__global__ void checkUnphysicalPsiKernel(const double* psi, bool* result, double D, double s, int stepNum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for reduction within a block
    __shared__ bool blockBool;
    if (threadIdx.x == 0) blockBool = false;
    __syncthreads();

    // Check condition
    if (idx < stepNum - 1) {
        double psiDiff = psi[idx] - psi[idx + 1];
        if (psiDiff > D || psiDiff < -2 * s) {
            atomicOr((int*)&blockBool, 1);  // Ensure updates are seen by all threads
        }
    }
    __syncthreads();

    // If any thread in the block found an issue, set the block result
    if (threadIdx.x == 0) {
        result[blockIdx.x] = blockBool;
    }
}

extern "C" bool checkUnphysicalPsiCUDA(double* psi, double D, double s, int stepNum) {
    int gridSize = (stepNum + blockSize - 1) / blockSize;
    bool* d_result;
    cudaMalloc(&d_result, gridSize * sizeof(bool));
    cudaMemset(d_result, 0, gridSize * sizeof(bool)); // Initialize to false

    // Launch kernel
    checkUnphysicalPsiKernel<<<gridSize, blockSize>>>(psi, d_result, D, s, stepNum);
    cudaDeviceSynchronize();

    // Copy result back to host
    std::vector<char> hostResults(gridSize);
    cudaMemcpy(hostResults.data(), d_result, gridSize * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    // If any block found an issue, return true
    for (bool found : hostResults) {
        if (found) return true;
    }
    return false;
}