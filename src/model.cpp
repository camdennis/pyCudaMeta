#include <iostream>
#include <vector>
#include "model.hpp"
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>

double pi = 3.14159265358979323846264338328;

// Declaration of pointwiseMultiply function (defined in model.cu)
extern "C" void updatePsiPosVelCUDA(double* psi, double* psiVel, double* psiForces, double m, double dt, int numPositions);
extern "C" void updatePsiForcesCUDA(double* psi, double* psiVel, double* psiLengths, double* psiForces, double k1, double g1, double a, double alpha, int numPositions);
extern "C" bool checkUnphysicalPsiCUDA(double* psi, double D, double s, int stepNum);
extern "C" void updatePhiPosVelCUDA(double* phi, double* phiVel, double* phiForces, double m, double dt, int numPositions);
extern "C" void updatePhiForcesCUDA(double* psi, double* phi, double* psiVel, double* phiVel, double* psiForces, double* phiForces, double k2, double g2, int numPositions);
extern "C" bool checkUnphysicalPhiCUDA(double* psi, double* phi, double R, double r, int stepNum);
extern "C" void updateVPosVelCUDA(double* psi, double* psiVel, double* v, double* vVel, double a, double D, double L, double s, int numPositions);
extern "C" void updateThetaPosVelCUDA(double* theta, double* thetaVel, double* thetaForces, double m3, double dt, int numPositions);
extern "C" void updateThetaForcesCUDA(double* v, double* vVel, double* theta, double* thetaVel, double* thetaForces, double* psiForces, double k3, double g3, double D, double L, int numPositions);

// Constructor
Model::Model(int size) : stringSize(size) {
    cudaFree(0);
    cudaMalloc((void**)&psi, stringSize * sizeof(double));
    cudaMalloc((void**)&psiVel, stringSize * sizeof(double));
    cudaMalloc((void**)&psiForces, stringSize * sizeof(double));
    cudaMalloc((void**)&psiLengths, stringSize * sizeof(double));
    cudaMalloc((void**)&phi, stringSize * sizeof(double));
    cudaMalloc((void**)&phiVel, stringSize * sizeof(double));
    cudaMalloc((void**)&v, stringSize * sizeof(double));
    cudaMalloc((void**)&vVel, stringSize * sizeof(double));
    cudaMalloc((void**)&theta, stringSize * sizeof(double));
    cudaMalloc((void**)&thetaVel, stringSize * sizeof(double));
    cudaMalloc((void**)&psiForces, stringSize * sizeof(double));
    cudaMalloc((void**)&phiForces, stringSize * sizeof(double));
    cudaMalloc((void**)&thetaForces, stringSize * sizeof(double));
}

void Model::setStringSize(const int stringSize_) {
    stringSize = stringSize_;
}

int Model::getStringSize() {
    return stringSize;
}

void Model::setParameters(const std::vector<double>& parameters) {
    k1 = parameters[0];
    k2 = parameters[1];
    k3 = parameters[2];
    m1 = parameters[3];
    m2 = parameters[4];
    m3 = parameters[5];
    L = parameters[6];
    D = parameters[7];
    s = sqrt(D * D + L * L) / 2.0;
    g1 = parameters[8];
    g2 = parameters[9];
    g3 = parameters[10];
    R = parameters[11];
    a = parameters[12];
    alpha = parameters[13];
    beta = parameters[14];
    gamma = parameters[15];
    r = parameters[16];
    dt = parameters[17];
}

std::vector<double> Model::getParameters() const {
    std::vector<double> parameters_(17);
    parameters_[0] = k1;
    parameters_[1] = k2;
    parameters_[2] = k3;
    parameters_[3] = m1;
    parameters_[4] = m2;
    parameters_[5] = m3;
    parameters_[6] = L;
    parameters_[7] = D;
    parameters_[8] = g1;
    parameters_[9] = g2;
    parameters_[10] = g3;
    parameters_[11] = R;
    parameters_[12] = a;
    parameters_[13] = alpha;
    parameters_[14] = beta;
    parameters_[15] = gamma;
    parameters_[16] = r;
    parameters_[17] = dt;
    return parameters_;
}

void Model::setDeathCut(double deathCut_) {
    deathCut = deathCut_;
}

void Model::setPsi(const std::vector<double>& psiData) {
    cudaMemcpy(psi, psiData.data(), stringSize * sizeof(double), cudaMemcpyHostToDevice);
}

std::vector<double> Model::getPsi() const {
    std::vector<double> psi_(stringSize);
    cudaMemcpy(psi_.data(), psi, stringSize * sizeof(double), cudaMemcpyDeviceToHost);
    return psi_;
}

std::vector<double> Model::getPsiLengths() const {
    std::vector<double> psiLengths_(stringSize);
    cudaMemcpy(psiLengths_.data(), psiLengths, stringSize * sizeof(double), cudaMemcpyDeviceToHost);
    return psiLengths_;
}

std::vector<double> Model::getPsiForces() const {
    std::vector<double> psiForces_(stringSize);
    cudaMemcpy(psiForces_.data(), psiForces, stringSize * sizeof(double), cudaMemcpyDeviceToHost);
    return psiForces_;
}

void Model::setPsiVel(const std::vector<double>& psiVelData) {
    cudaMemcpy(psiVel, psiVelData.data(), stringSize * sizeof(double), cudaMemcpyHostToDevice);
}

std::vector<double> Model::getPsiVel() const {
    std::vector<double> psiVel_(stringSize);
    cudaMemcpy(psiVel_.data(), psiVel, stringSize * sizeof(double), cudaMemcpyDeviceToHost);
    return psiVel_;
}

void Model::setPhi(const std::vector<double>& phiData) {
    cudaMemcpy(phi, phiData.data(), stringSize * sizeof(double), cudaMemcpyHostToDevice);
}

std::vector<double> Model::getPhi() const {
    std::vector<double> phi_(stringSize);
    cudaMemcpy(phi_.data(), phi, stringSize * sizeof(double), cudaMemcpyDeviceToHost);
    return phi_;
}

void Model::setPhiVel(const std::vector<double>& phiVelData) {
    cudaMemcpy(phiVel, phiVelData.data(), stringSize * sizeof(double), cudaMemcpyHostToDevice);
}

std::vector<double> Model::getPhiVel() const {
    std::vector<double> phiVel_(stringSize);
    cudaMemcpy(phiVel_.data(), phiVel, stringSize * sizeof(double), cudaMemcpyDeviceToHost);
    return phiVel_;
}

void Model::setTheta(const std::vector<double>& thetaData) {
    cudaMemcpy(theta, thetaData.data(), stringSize * sizeof(double), cudaMemcpyHostToDevice);
}

std::vector<double> Model::getTheta() const {
    std::vector<double> theta_(stringSize);
    cudaMemcpy(theta_.data(), theta, stringSize * sizeof(double), cudaMemcpyDeviceToHost);
    return theta_;
}

void Model::setThetaVel(const std::vector<double>& thetaVelData) {
    cudaMemcpy(thetaVel, thetaVelData.data(), stringSize * sizeof(double), cudaMemcpyHostToDevice);
}

std::vector<double> Model::getThetaVel() const {
    std::vector<double> thetaVel_(stringSize);
    cudaMemcpy(thetaVel_.data(), thetaVel, stringSize * sizeof(double), cudaMemcpyDeviceToHost);
    return thetaVel_;
}

std::vector<double> Model::getV() const {
    std::vector<double> v_(stringSize);
    cudaMemcpy(v_.data(), v, stringSize * sizeof(double), cudaMemcpyDeviceToHost);
    return v_;
}

std::vector<double> Model::getVVel() const {
    std::vector<double> vVel_(stringSize);
    cudaMemcpy(vVel_.data(), vVel, stringSize * sizeof(double), cudaMemcpyDeviceToHost);
    return vVel_;
}

void Model::updatePsiForces(int stepNum) {
//    cudaMemset(psiForces, 0.0, stringSize * sizeof(double));
    if (stepNum == -1) {
        stepNum = stringSize;
    }
    else {
        stepNum = std::min(stringSize, stepNum);
    }
    updatePsiForcesCUDA(psi, psiVel, psiLengths, psiForces, k1, g1, a, alpha, stepNum);
}

void Model::updatePsiPosVel(int stepNum) {
    if (stepNum == -1) {
        stepNum = stringSize;
    }
    else {
        stepNum = std::min(stringSize, stepNum);
    }
    updatePsiPosVelCUDA(psi, psiVel, psiForces, m1, dt, stepNum);
}

bool Model::checkUnphysicalPsi(int stepNum) {
    if (stepNum == -1) {
        stepNum = stringSize;
    }
    else {
        stepNum = std::min(stringSize, stepNum);
    }
    return checkUnphysicalPsiCUDA(psi, D, s, stepNum);
}

void Model::updatePhiForces(int stepNum) {
//    cudaMemset(psiForces, 0.0, stringSize * sizeof(double));
    if (stepNum == -1) {
        stepNum = stringSize;
    }
    else {
        stepNum = std::min(stringSize, stepNum);
    }
    updatePhiForcesCUDA(psi, phi, psiVel, phiVel, psiForces, phiForces, k2, g2, stepNum);
}

void Model::updatePhiPosVel(int stepNum) {
    if (stepNum == -1) {
        stepNum = stringSize;
    }
    else {
        stepNum = std::min(stringSize, stepNum);
    }
    updatePhiPosVelCUDA(phi, phiVel, phiForces, m2, dt, stepNum);
}

void Model::updateVPosVel(int stepNum) {
    if (stepNum == -1) {
        stepNum = stringSize;
    }
    else {
        stepNum = std::min(stringSize, stepNum);
    }
    updateVPosVelCUDA(psi, psiVel, v, vVel, a, D, L, s, stepNum);
}

void Model:: updateThetaForces(int stepNum) {
    updateThetaForcesCUDA(v, vVel, theta, thetaVel, thetaForces, psiForces, k3, g3, D, L, stepNum);
}

void Model::updateThetaPosVel(int stepNum) {
    updateThetaPosVelCUDA(theta, thetaVel, thetaForces, m3, dt, stepNum);
}

bool Model::checkUnphysicalPhi(int stepNum) {
    if (stepNum == -1) {
        stepNum = stringSize;
    }
    else {
        stepNum = std::min(stringSize, stepNum);
    }
    return checkUnphysicalPhiCUDA(psi, phi, R, r, stepNum);
}

void Model::drive(double amplitude, double omega, double t) {
    double arg = (omega * t);
    int overshoot = arg / (2 * pi);
    arg -= overshoot * 2 * pi;
    double val = amplitude * sin(arg);
    double valV = amplitude * omega * cos(arg);
    cudaMemcpy(psi + 0, &val, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(psiVel + 0, &valV, sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

bool Model::checkDeathCut() {
    double psiEnd;
    cudaMemcpy(&psiEnd, psi + stringSize - 1, sizeof(double), cudaMemcpyDeviceToHost);
    if (psiEnd > deathCut) {
        return true;
    }
    return false;
}

int Model::runSimulation(double amplitude, double omega, double T) {
    int numSteps = T / dt;
    for (int i = 0; i < numSteps; i++) {
        updatePsiForces(i + 1);
        updatePhiForces(i + 1);
        updateThetaForces(i + 1);
        updatePsiPosVel(i + 1);
        updatePhiPosVel(i + 1);
        updateVPosVel(i + 1);
        updateThetaPosVel(i + 1);
        drive(amplitude, omega, i * dt);
        if (i > stringSize) {
            if (checkDeathCut()) {
                return dt * i;
            }
        }
        if (checkUnphysicalPsi(i + 1) || checkUnphysicalPhi(i + 1)) {
            return dt * i;
        }
    }
    return numSteps * dt;
}