#include <cuda_runtime.h>
#include <iostream>
#include <float.h>

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