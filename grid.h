#ifndef GRID_H
#define GRID_H

#include <vector>
#include <iostream>
#include <utility>
#include <stdexcept>
#include <cmath>

namespace gridmthd {

    typedef std::vector<std::vector<std::vector<std::pair<double, double>>>> Grid;
    
    Grid buildGrid(std::vector<double> x, std::vector<double> y);

    std::pair<std::vector<double>, std::vector<double>> calculateDistancesWithGrid(Grid grid);
}

#endif