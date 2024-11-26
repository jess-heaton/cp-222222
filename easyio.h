#ifndef EASYIO_H
#define EASYIO_H

#include <vector>
#include <iostream>

namespace utils {
    namespace easyio {
        void readCsv(const std::string& fileName, std::vector<std::vector<double>>& data);
        void outputToTxt(const std::string filename, const std::vector<double>& data);
    }
}

#endif