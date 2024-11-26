#include <cassert>
#include "rng.h"

namespace utils {
    namespace rand {
        // The definition of variable rng, declared in rng.h
        Rng rng(static_cast<unsigned>(std::random_device()()));

        // The definitions of the functions declared in rng.h
        void seedRand(uint_fast32_t seed) {
            rng.seed(seed);
        }

        int randInt(int rangeBegin, int rangeEnd) {
            // Function to produce a random integer in [rangeBegin, rangeEnd).
            assert(rangeEnd > rangeBegin);
            std::uniform_int_distribution<> dist {rangeBegin, rangeEnd - 1};
            return dist(rng);
        }

        double randDouble(double rangeBegin, double rangeEnd) {
            // Function to produce a random double in [rangeBegin, rangeEnd).
            std::uniform_real_distribution<> dist {rangeBegin, rangeEnd};
            return dist(rng);
        }
    }
}