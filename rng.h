#ifndef RNG_H
#define RNG_H

#include <random>
#include <utility>

namespace utils {
    namespace rand {
        // Rng is an alias for our chosen type of random number generator.
        // Here we have chosen std::mt19937: a ’Mersenne twister’.
        using Rng = std::mt19937;

        // A variable declaration. Starting with ’extern’, this tells the
        // compiler that a variable rng, of type Rng, is defined somewhere.
        extern Rng rng;

        // Function declarations. Again, these tell the compiler that these
        // functions are defined somewhere.
        void seedRand(uint_fast32_t seed);
        
        int randInt(int rangeBegin, int rangeEnd);
        
        double randDouble(double rangeBegin, double rangeEnd);
    }
}

#endif