#include <vector>
#include <iostream>
#include <utility>
#include <omp.h>
#include <cmath>


using namespace std;

namespace gridmthd {

    typedef vector<vector<vector<pair<double, double>>>> Grid;

    Grid buildGrid(vector<double> x, vector<double> y) {
        size_t n = x.size();
        
        // for calculating furthests we need that the furthest point will be in one of the corners almost surely
        // so pts per cell needs to be ~100
        // do some probability stuff on this to determine how many points this needs for us to be sure of it
        int ptsPerCell = 100;
        int cellCount = n / ptsPerCell;
        int groupsPerAxis = sqrt(cellCount);

        double cellSize = 1 / sqrt(cellCount); 
        
        // initialise how big the grid should be
        Grid grid(groupsPerAxis, vector<vector<pair<double, double>>>(groupsPerAxis, vector<pair<double, double>>()));

        for (size_t i = 0; i < n; ++i) {
            // use emplace to build the pair inside the vector, a bit faster
            int xInd = min(groupsPerAxis - 1, (int)(x[i] / cellSize));
            int yInd = min(groupsPerAxis - 1, (int)(y[i] / cellSize));
            
            grid[xInd][yInd].emplace_back(x[i], y[i]);        
        }

        return grid;
    }

    // Assumes that all cells in the grid have at least one point
    pair<vector<double>, vector<double>> calculateDistancesWithGrid(Grid grid) {
        int cellCountPerD = grid.size();

        if (cellCountPerD != grid[0].size()) {
            throw invalid_argument("x and y must have the same number of elements");
        }
        
        // establish edge cells for furthest calculations (avoiding duplicates for corners)
        vector<pair<int, int>> edgeIndices = {
            {0, 0}, {0, cellCountPerD-1}, {cellCountPerD-1, 0}, {cellCountPerD-1, cellCountPerD-1} 
        };
        
        for (int i = 1; i < cellCountPerD - 1; ++i) {
            edgeIndices.emplace_back(i, 0);
            edgeIndices.emplace_back(0, i);
            edgeIndices.emplace_back(cellCountPerD-1, i);
            edgeIndices.emplace_back(i, cellCountPerD-1);
        }
        
        // calculate nearests
        vector<double> nearests = {};
        vector<double> furthests = {};
        
        // iter through all grids
        for (int i = 0; i < cellCountPerD; ++i) {
            for (int j = 0; j < cellCountPerD; ++j) {
                
                if (grid[i][j].empty()) continue; // skip empty cells

                
                for (int k = 0; k < grid[i][j].size(); ++k) {
                
                    // iter through all points
                    double runningNearest = 1.5; // arbitrary number > sqrt(2)
                    
                    // find nearest by going through relevant grids s.t. we check all neighbour grids
                    for (int l = max(0, i-1); l <= min(cellCountPerD-1, i+1); ++l) {
                        for (int m = max(0, j-1); m <= min(cellCountPerD-1, j+1); ++m) {
                            
                            // go through points in neighboring cells
                            for (int n = 0; n < grid[l][m].size(); ++n) {
                                
                                // skip comparing point with itself
                                if (l == i && m == j && n == k) {
                                    continue;
                                }
                                
                                double xdiff = grid[i][j][k].first - grid[l][m][n].first;
                                double ydiff = grid[i][j][k].second - grid[l][m][n].second;
                                double disSquared = xdiff * xdiff + ydiff * ydiff;
                                
                                runningNearest = min(runningNearest, disSquared);
                            }
                        }
                    }   

                    nearests.push_back(sqrt(runningNearest));

                    // find furthest by checking the edge cells
                    double runningFurthest = -1; // arbitrary number < 0

                    for (auto cellInd : edgeIndices) {
                        
                        // go through points in that cell
                        for (int l = 0; l < grid[cellInd.first][cellInd.first].size(); ++l) {
                            
                            double xdiff = grid[i][j][k].first - grid[cellInd.first][cellInd.second][l].first;
                            double ydiff = grid[i][j][k].second - grid[cellInd.first][cellInd.second][l].second;
                            
                            double distSquared = xdiff * xdiff + ydiff * ydiff;
                            
                            runningFurthest = max(runningFurthest, distSquared);
                        }
                    }

                    furthests.push_back(sqrt(runningFurthest));
                }
            }
        }    
        
        return {nearests, furthests};
    }   
}