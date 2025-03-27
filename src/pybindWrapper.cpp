#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // This is needed for std::vector binding
#include "model.hpp"
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(libpyCudaMeta, m) {

    py::class_<Model>(m, "Model")
        .def(py::init<int>())
        // Setters
        .def("setPsi", &Model::setPsi)
        .def("setPsiVel", &Model::setPsiVel)
        .def("setStringSize", &Model::setStringSize)
        .def("setParameters", &Model::setParameters)
        .def("setDeathCut", &Model::setDeathCut)

        // Doers
        .def("updatePsiForces", &Model::updatePsiForces)
        .def("updatePsiPosVel", &Model::updatePsiPosVel)
        .def("runSimulation", &Model::runSimulation)
        .def("drive", &Model::drive)
        .def("checkUnphysicalPsi", &Model::checkUnphysicalPsi)

        // Getters
        .def("getStringSize", &Model::getStringSize)
        .def("getParameters", &Model::getParameters)
        .def("getPsi", &Model::getPsi)
        .def("getPsiLengths", &Model::getPsiLengths)
        .def("getPsiForces", &Model::getPsiForces);
}
