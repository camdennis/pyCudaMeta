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

__global__ void updatePhiPosVelKernel(double* phi, double* phiVel, double* phiForces, double m, double dt, int numPositions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPositions) {
        phi[idx] += phiVel[idx] * dt + dt * dt * phiForces[idx] / 2.0 / m;
        phiVel[idx] += phiForces[idx] * dt / m;
    }
}

__global__ void updatePhiForcesKernel(double* psi, double* phi, double* psiVel, double* phiVel, double* psiForces, double* phiForces, double k2, double g2, int numPositions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPositions) {
        double force = k2 * (psi[idx] - phi[idx]) + g2 * (psiVel[idx] - phiVel[idx]);
        phiForces[idx] = force;
        psiForces[idx] -= force;
    }
}

__global__ void checkUnphysicalPhiKernel(const double* psi, const double* phi, bool* result, double R, double r, int stepNum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Shared memory for reduction within a block
    __shared__ bool blockBool;
    if (threadIdx.x == 0) blockBool = false;
    __syncthreads();

    // Check condition
    if (idx < stepNum - 1) {
        double diff = abs(psi[idx] - phi[idx]);
        if (diff > R - r) {
            atomicOr((int*)&blockBool, 1);  // Ensure updates are seen by all threads
        }
    }
    __syncthreads();

    // If any thread in the block found an issue, set the block result
    if (threadIdx.x == 0) {
        result[blockIdx.x] = blockBool;
    }
}

__global__ void updateThetaPosVelKernel(double* theta, double* thetaVel, double* thetaForces, double m3, double dt, int numPositions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPositions) {
        theta[idx] += thetaVel[idx] * dt + dt * dt * thetaForces[idx] / 2.0 / m3;
        thetaVel[idx] += thetaForces[idx] * dt / m3;
    }
}

__global__ void updateThetaForcesKernel(double* v, double* vVel, double* theta, double* thetaVel, double* thetaForces, double* psiForces, double k3, double g3, double D, double L, int numPositions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPositions - 1) {
        double force = k3 * (v[idx] - theta[idx]) + g3 * (vVel[idx] - thetaVel[idx]);
        double forceOnPsi = -force * sqrt(D * D - 4 * L * v[idx] - 4 * v[idx] * v[idx]) / (2 * v[idx] + L);
        thetaForces[idx] = force;
        psiForces[idx] += forceOnPsi;
    }
}

__global__ void updateVPosVelKernel(double* psi, double* psiVel, double* v, double* vVel, double a, double D, double L, double s, int numPositions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPositions - 1) {
        double denom = sqrt(4 * s * s - D * D + 2 * D * (psi[idx] - psi[idx + 1]) - (psi[idx] - psi[idx + 1]) * (psi[idx] - psi[idx + 1]));
        v[idx] = -L / 2.0 + denom / 2.0;
        vVel[idx] = (D - psi[idx] + psi[idx + 1]) * (psiVel[idx] - psiVel[idx + 1]);
        vVel[idx] /= (2.0 * denom);
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

extern "C" void updatePhiPosVelCUDA(double* phi, double* phiVel, double* phiForces, double m, double dt, int numPositions) {
    int gridSize = (numPositions + blockSize - 1) / blockSize;
    updatePhiPosVelKernel<<<gridSize, blockSize>>>(phi, phiVel, phiForces, m, dt, numPositions);
    cudaDeviceSynchronize();
}

extern "C" void updatePhiForcesCUDA(double* psi, double* phi, double* psiVel, double* phiVel, double* psiForces, double* phiForces, double k2, double g2, int numPositions) {
    int gridSize = (numPositions + blockSize - 1) / blockSize;
    updatePhiForcesKernel<<<gridSize, blockSize>>>(psi, phi, psiVel, phiVel, psiForces, phiForces, k2, g2, numPositions);
    cudaDeviceSynchronize();
}

extern "C" bool checkUnphysicalPhiCUDA(double* psi, double* phi, double R, double r, int stepNum) {
    int gridSize = (stepNum + blockSize - 1) / blockSize;
    bool* d_result;
    cudaMalloc(&d_result, gridSize * sizeof(bool));
    cudaMemset(d_result, 0, gridSize * sizeof(bool)); // Initialize to false

    // Launch kernel
    checkUnphysicalPhiKernel<<<gridSize, blockSize>>>(psi, phi, d_result, R, r, stepNum);
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

extern "C" void updateVPosVelCUDA(double* psi, double* psiVel, double* v, double* vVel, double a, double D, double L, double s, int numPositions) {
    int gridSize = (numPositions + blockSize - 1) / blockSize;
    updateVPosVelKernel<<<gridSize, blockSize>>>(psi, psiVel, v, vVel, a, D, L, s, numPositions);
    cudaDeviceSynchronize();
}

extern "C" void updateThetaPosVelCUDA(double* theta, double* thetaVel, double* thetaForces, double m3, double dt, int numPositions) {
    int gridSize = (numPositions + blockSize - 1) / blockSize;
    updateThetaPosVelKernel<<<gridSize, blockSize>>>(theta, thetaVel, thetaForces, m3, dt, numPositions);
    cudaDeviceSynchronize();
}

extern "C" void updateThetaForcesCUDA(double* v, double* vVel, double* theta, double* thetaVel, double* thetaForces, double* psiForces, double k3, double g3, double D, double L, int numPositions) {
    int gridSize = (numPositions + blockSize - 1) / blockSize;
    updateThetaForcesKernel<<<gridSize, blockSize>>>(v, vVel, theta, thetaVel, thetaForces, psiForces, k3, g3, D, L, numPositions);
    cudaDeviceSynchronize();
}
