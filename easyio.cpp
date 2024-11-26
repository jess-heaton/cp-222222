#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>

#include "easyio.h"

namespace utils {
    namespace easyio {

        // Reads a csv of format
        // 0.2, 0.9
        // 0.3, 0.5
        // 0.2, 0.7
        // ...
        // 
        // and writes by reference to a given data variable. 
        // Remove header file from csv before using. 
        void readCsv(const std::string& fileName, std::vector<std::vector<double>>& data) {
            // open file
            std::ifstream file(fileName);
            if (!file.is_open()) {
                // eg user doesn't have perms to open it
                std::cerr << "err: could not open file " << fileName << std::endl;
                exit(1);
            }

            std::string line;
            std::vector<double> xCoords, yCoords;

            // keep reading from the csv until it fails to give us another line
            while (std::getline(file, line)) {
                std::istringstream lineIn(line);
                double x, y;
                char comma;

                if (lineIn >> x >> comma >> y && comma == ',') {
                    xCoords.push_back(x);
                    yCoords.push_back(y);
                } else {
                    std::cerr << "err: malformed line in csv" << line << std::endl;
                    exit(1);
                }
            }

            file.close();

            // assign to output data
            data.push_back(xCoords);
            data.push_back(yCoords);
        }

        void outputToTxt(const std::string filename, const std::vector<double>& data) {
            std::ofstream out(filename);
            if (!out.is_open()) {
                std::cerr << "Failed to open file for writing." << std::endl;
            }

            for (int i = 0; i < data.size(); ++i) {
                out << data[i] << std::endl;
            }
        }
    }
}