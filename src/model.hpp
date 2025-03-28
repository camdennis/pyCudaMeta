#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include <cuda_runtime.h>
#include <math.h>

class Model {
public:
    Model(int size);             
    // Constructor declaration
    void setDeathCut(double deathCut);
    bool checkDeathCut();
    void setStringSize(int stringSize);
    int getStringSize();
    void setPsi(const std::vector<double> &data);
    void setPsiVel(const std::vector<double>& data);
    void setPhi(const std::vector<double>& data);
    void setPhiVel(const std::vector<double>& data);
    void setTheta(const std::vector<double>& data);
    void setThetaVel(const std::vector<double>& data);
    void setParameters(const std::vector<double>& data);
    void drive(double amplitude, double omega, double t);
    int runSimulation(double amplitude, double omega, double T);
    std::vector<double> getPsi() const;  // Return the result matrix
    std::vector<double> getPsiVel() const;  // Return the result matrix
    std::vector<double> getPsiLengths() const;  // Return the result matrix
    std::vector<double> getPsiForces() const;  // Return the result matrix
    std::vector<double> getPhi() const;  // Return the result matrix
    std::vector<double> getPhiVel() const;  // Return the result matrix
    std::vector<double> getTheta() const;  // Return the result matrix
    std::vector<double> getThetaVel() const;  // Return the result matrix
    std::vector<double> getV() const;  // Return the result matrix
    std::vector<double> getVVel() const;  // Return the result matrix
    std::vector<double> getParameters() const;
    bool checkUnphysicalPsi(int stepNum);
    void updatePsiForces(int stepNum);
    void updatePsiPosVel(int stepNum);

    bool checkUnphysicalPhi(int stepNum);
    void updatePhiForces(int stepNum);
    void updatePhiPosVel(int stepNum);
    void updateVPosVel(int stepNum);

    void updateThetaForces(int stepNum);
    void updateThetaPosVel(int stepNum);

private:
    int stringSize;
    double deathCut = 0.0;
    double* psi;
    double* psiVel;
    double* phi;
    double* phiVel;
    double* v;
    double* vVel;
    double* theta;
    double* thetaVel;
    double* psiForces;
    double* phiForces;
    double* thetaForces;
    double* psiLengths;
    double k1, k2, k3, m1, m2, m3, L, D, s, g1, g2, g3, R, a, alpha, beta, gamma, r;
    double dt;
};

#endif
