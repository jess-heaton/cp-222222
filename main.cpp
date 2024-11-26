#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <utility>
#include <omp.h>
#include <iomanip> // Required for setprecision


#include "easyio.h"
#include "rng.h"
#include "grid.h"

// small program, no danger of namespace pollution. Makes code more readable, good to do.
using namespace std; 

// taken from https://stackoverflow.com/questions/22387586/measuring-execution-time-of-a-function-in-c
// changed slightly to give result in seconds rather than nanoseconds
typedef chrono::high_resolution_clock::time_point TimeVar;
#define duration(a) chrono::duration<double>(a).count()
#define timeNow() chrono::high_resolution_clock::now()

template<typename F, typename... Args>
void funcTime(F func, Args&&... args){
    TimeVar t1=timeNow();
    
    func(std::forward<Args>(args)...);
    cout << "time taken: " << duration(timeNow()-t1)
         << "\n\n"; // dbl new line to make results more readable
}
// end stackoverflow reference

// write extended results to plaintext files, and print titled averages.
void printResults(
    vector<double> &nearests, vector<double> &furthests, double avgNearest, double avgFurthest,
    bool isParallel, bool isWraparoundGeo
) {
    // write whole list to txt files
    utils::easyio::outputToTxt("basicNearests.txt", nearests);
    utils::easyio::outputToTxt("basicFurthests.txt", furthests);

    string title = "Calculating in " + string(isParallel ? "parallel " : "serial ") +
                   "using " + string(isWraparoundGeo ? "wraparound geometry" : "basic geometry");

    // print averages to stdout
    cout << title << endl
         << "mean nearest distance: " << avgNearest << endl
         << "mean furthest distance: " << avgFurthest << endl;
}

// Returns 100k randomly initialised points, from 0.0 to 1.0 in the x & y direction. 
// In the format of {{x_1, x_2, ..., x_n},  {y_1, y_2, ..., x_n}}
// int n : the number of points to create
vector<vector<double>> randomPoints(int n) {
    double start = 0;
    double end = 1;
    
    vector<double> x(n, start);
    vector<double> y(n, start);

    for (int i = 0; i < n; ++i) {
        // gen a random x and y coord
        
        x[i] = utils::rand::randDouble(start, end);
        y[i] = utils::rand::randDouble(start, end);
    }

    return vector<vector<double>> {x, y};
}

// Calculates the shortest and longest distance, given the x and y distances, between two points.
// Places the shortest distance in arr[0], the longest distance in arr[1]. 
// Returns the distances squared. 
// 
// arr[0] must have size >= 2.
// Uses wraparound geometry.
void wraparoundGeo(double xdiff, double ydiff, double arr[]) {
    double dy = min(ydiff, 1 - ydiff);
    double dx = min(xdiff, 1 - xdiff);
    
    double closestDist = dy * dy + dx * dx;

    dy = max(ydiff, 1 - ydiff);
    dx = max(xdiff, 1 - xdiff);

    double furthestDist = dy * dy + dx * dx;

    arr[0] = closestDist;
    arr[1] = furthestDist;
}

// Calculates the Euclidean distance between two points, given the x and y distances between them.
// The distance is placed in both arr[0] and arr[1]. 
// Returns the distances squared. 
// 
// arr[0] must have size >= 2.
void basicGeo(double xdiff, double ydiff, double arr[]) {
    double dist = ydiff * ydiff + xdiff * xdiff;
    
    arr[0] = dist;
    arr[1] = dist;
}

// For each given point (x_i, y_i), calculate the distance to the nearest and furthest point.
// For calcFunc, pass the function that describes the geometry to use. Basic and wraparound are
// both given in this file.
// 
// Nearest points will be written to nearest.txt, furthests to furthest.txt. 
// Prints to stdout the average nearest and furthest distances.
// Calculated in serial.
void calcNearestAndFurthestDistances_Serial(
    vector<double> x, vector<double> y, void (*calcFunc)(double, double, double[])
) {
    int n = x.size();

    if (n != y.size()) {
        throw invalid_argument("x and y must have the same number of elements");
    }
    
    // setup
    vector<double> nearests(n, 0);
    vector<double> furthests(n, 0);
    
    // contains {nearest dist, furthest dist}. No need to reinitialise ever, just reuse it.
    double res[2];
    
    // iter through all points
    for (int i = 0; i < n; ++i) {
        // calc for each point
        double runningNearestSquared = 2.1; // arbitrary number > sqrt(2) ** 2 = 2
        double runningFurthestSquared = -0.1; // arbitrary number < 0
        
        for (int j = 0; j < n; ++j) {
            // calculate euclidean distances in x & y
            double ydiff = abs(y[j] - y[i]);
            double xdiff = abs(x[j] - x[i]);

            // pass to calcFunc to compute nearest and furthest distances, basic dependency injection
            calcFunc(xdiff, ydiff, res);
            
            if (res[0] < runningNearestSquared && i != j) {
                runningNearestSquared = res[0];
            }
            if (res[1] > runningFurthestSquared && i != j) {
                runningFurthestSquared = res[1];
            }
        }

        nearests[i] = sqrt(runningNearestSquared);
        furthests[i] = sqrt(runningFurthestSquared);
    }

    double nearestMean = 0;
    double furthestMean = 0;
    for (int i = 0; i < n; ++i) {
        nearestMean += nearests[i];
        furthestMean += furthests[i];
    }
    double avgNearest = nearestMean / n;
    double avgFurthest = furthestMean / n;

    // output results
    printResults(nearests, furthests, avgNearest, avgFurthest, false, calcFunc == wraparoundGeo);
}

// For each given point (x_i, y_i), calculate the distance to the nearest and furthest point.
// For calcFunc, pass the function that describes the geometry to use. Basic and wraparound are
// both given in this file.
// 
// Nearest points will be written to nearest.txt, furthests to furthest.txt. 
// Prints to stdout the average nearest and furthest distances.
// Calculated using parallelisation from OpenMP.
void calcNearestAndFurthestDistances_Parallel(
    vector<double> x, vector<double> y, void (*calcFunc)(double, double, double[])
) {
    int n = x.size();

    if (n != y.size()) {
        throw invalid_argument("x and y must have the same number of elements");
    }
    
    // setup
    vector<double> nearests(n, 0);
    vector<double> furthests(n, 0);

    // contains {nearest dist, furthest dist}. No need to reinitialise ever, just reuse it.
    double res[2];
    
    // iter through all points
    #pragma omp parallel for private(res) // make sure that each thread has its own threadbound`res`
    for (int i = 0; i < n; ++i) {
        // calc for each point
        double runningNearestSquared = 2.1; // arbitrary number > sqrt(2) ** 2 = 2
        double runningFurthestSquared = -0.1; // arbitrary number < 0

        for (int j = 0; j < n; ++j) {
            // calculate euclidean distances in x & y
            double ydiff = abs(y[j] - y[i]);
            double xdiff = abs(x[j] - x[i]);

            // pass to calcFunc to compute nearest and furthest distances, basic dependency injection
            calcFunc(xdiff, ydiff, res);
            
            if (res[0] < runningNearestSquared && i != j) {
                runningNearestSquared = res[0];
            }
            if (res[1] > runningFurthestSquared && i != j) {
                runningFurthestSquared = res[1];
            }
        }

        nearests[i] = sqrt(runningNearestSquared);
        furthests[i] = sqrt(runningFurthestSquared);
    }

    // calculate means
    double nearestSum = 0;
    double furthestSum = 0;
    #pragma omp parallel for reduction(+:nearestSum, furthestSum) // prevent race conditions 
    for (int i = 0; i < n; ++i) {
        nearestSum += nearests[i];
        furthestSum += furthests[i];
    }
    
    double avgNearest = nearestSum / n;
    double avgFurthest = furthestSum / n;

    // output results
    printResults(nearests, furthests, avgNearest, avgFurthest, true, calcFunc == wraparoundGeo);
}

// For each given point (x_i, y_i), calculate the distance to the nearest and furthest point.
// For calcFunc, pass the function that describes the geometry to use. Basic and wraparound are
// both given in this file.
// 
// Nearest points will be written to nearest.txt, furthests to furthest.txt. 
// Prints to stdout the average nearest and furthest distances.
// Calculated in serial.
// Uses the faster algorithm.
void calcNearestAndFurthestDistances_Serial_Fast(
    vector<double> x, vector<double> y, void (*calcFunc)(double, double, double[])
) {
    int n = x.size();

    if (n != y.size()) {
        throw invalid_argument("x and y must have the same number of elements");
    }
    
    // setup
    vector<double> nearests(n, 1.5);
    vector<double> furthests(n, -0.1);
    
    // contains {nearest dist, furthest dist}. No need to reinitialise ever, just reuse it.
    double res[2];
    
    // iter through all points
    for (int i = 0; i < n; ++i) {
        // calc for each pair (i, j)
        
        for (int j = i + 1; j < n; ++j) {
            // calculate euclidean distances in x & y
            double ydiff = abs(y[j] - y[i]);
            double xdiff = abs(x[j] - x[i]);

            // pass to calcFunc to compute nearest and furthest distances, basic dependency injection
            calcFunc(xdiff, ydiff, res);
            
            // update min & max for point i
            if (res[0] < nearests[i]) {
                nearests[i] = res[0];
            }
            if (res[1] > furthests[i]) {
                furthests[i] = res[1];
            }

            // update min & max for point j
            if (res[0] < nearests[j]) {
                nearests[j] = res[0];
            }
            if (res[1] > furthests[j]) {
                furthests[j] = res[1];
            }
        }
    }

    double nearestMean = 0;
    double furthestMean = 0;
    for (int i = 0; i < n; ++i) {
        nearests[i] = sqrt(nearests[i]);
        furthests[i] = sqrt(furthests[i]);

        nearestMean += nearests[i];
        furthestMean += furthests[i];
    }
    double avgNearest = nearestMean / n;
    double avgFurthest = furthestMean / n;

    // output results
    printResults(nearests, furthests, avgNearest, avgFurthest, false, calcFunc == wraparoundGeo);
}

// For each given point (x_i, y_i), calculate the distance to the nearest and furthest point.
// For calcFunc, pass the function that describes the geometry to use. Basic and wraparound are
// both given in this file.
// 
// Nearest points will be written to nearest.txt, furthests to furthest.txt. 
// Prints to stdout the average nearest and furthest distances.
// Calculated in parallel.
// Uses the faster algorithm.
void calcNearestAndFurthestDistances_Parallel_Fast(
    vector<double> x, vector<double> y, void (*calcFunc)(double, double, double[])
) {
    int n = x.size();

    if (n != y.size()) {
        throw invalid_argument("x and y must have the same number of elements");
    }
    
    // setup
    vector<double> nearests(n, 1.5);
    vector<double> furthests(n, -0.1);
    
    
    // iter through all points
    #pragma omp parallel
    {
        // contains {nearest dist, furthest dist}. 
        // Thread local variable
        double res[2];
        
        #pragma omp for schedule(dynamic, 32) // dynamic over guided, explained in report
        for (int i = 0; i < n; ++i) {
            // calc for each pair (i, j)
        
            
            for (int j = i + 1; j < n; ++j) {
                // calculate euclidean distances in x & y
                double ydiff = abs(y[j] - y[i]);
                double xdiff = abs(x[j] - x[i]);

                // pass to calcFunc to compute nearest and furthest distances, basic dependency injection
                calcFunc(xdiff, ydiff, res);
                
                // update min & max for point i
                // check once outside the critical region, which could be a stale read (ie, there could be
                // another thread updating this with an even closer / even further distance)
                // then check again once inside the critical region so that we get a guaranteed read
                if (res[0] < nearests[i]) {
                    #pragma omp critical
                    {
                        if (res[0] < nearests[i]) {
                            nearests[i] = res[0];
                        }
                    }
                }
                if (res[1] > furthests[i]) {
                    #pragma omp critical
                    {
                        if (res[1] > furthests[i]) {
                            furthests[i] = res[1];
                        }
                    }
                }

                // do the same for point j
                if (res[0] < nearests[j]) {
                    #pragma omp critical
                    {
                        if (res[0] < nearests[j]) { 
                            nearests[j] = res[0];
                        }
                    }
                }
                if (res[1] > furthests[j]) {
                    #pragma omp critical
                    {
                        if (res[1] > furthests[j]) { 
                            furthests[j] = res[1];
                        }
                    }
                }
            }
        }
    }

    double nearestMean = 0;
    double furthestMean = 0;
    
    #pragma omp parallel for reduction(+:nearestMean,furthestMean)
    for (int i = 0; i < n; ++i) {
        nearests[i] = sqrt(nearests[i]);
        furthests[i] = sqrt(furthests[i]);
        
        nearestMean += nearests[i];
        furthestMean += furthests[i];
    }
    double avgNearest = nearestMean / n;
    double avgFurthest = furthestMean / n;

    // output results
    printResults(nearests, furthests, avgNearest, avgFurthest, true, calcFunc == wraparoundGeo);
}

// Uses the grid method defined in grid.cpp to calculate nearests
void calculateNearestsWithGridMethod(
    vector<double> x, vector<double> y
) {
    int n = x.size();
    
    gridmthd::Grid grid = gridmthd::buildGrid(x, y);
    pair<vector<double>, vector<double>> res = gridmthd::calculateDistancesWithGrid(grid);

    double nearestMean = 0;
    double furthestMean = 0;
    for (int i = 0; i < n; ++i) {
        nearestMean += res.first[i];
        furthestMean += res.second[i];
    }

    double avgNearest = nearestMean / n;
    double avgFurthest = furthestMean / n;

    cout << "mean nearest: " << avgNearest << " mean furthest: " << avgFurthest << endl;
}

int main () {
    omp_set_num_threads(10);
    cout << fixed << setprecision(10); // print doubles with 10 decimal places
    
    int n = 100000;
    
    vector<vector<double>> pointsRandom = randomPoints(n);

    vector<vector<double>> pointsCSV;
    utils::easyio::readCsv("100000 locations.csv", pointsCSV);

    // examples
    
    // // SERIAL
    // // n randomly-initialised points, using basic geometry.
    //     funcTime(
      //       calcNearestAndFurthestDistances_Serial, pointsRandom[0], pointsRandom[1], basicGeo
    //     );
    // // n randomly-initialised points, using wraparound geometry.
    //     funcTime(
    //         calcNearestAndFurthestDistances_Serial, pointsRandom[0], pointsRandom[1], wraparoundGeo
    //     );
    // // n csv-initialised points, using basic geometry.
    //     funcTime(
    //         calcNearestAndFurthestDistances_Serial, pointsCSV[0], pointsCSV[1], basicGeo
    //     );

    // // PARALLELISATION
    // // n randomly-initialised points, using basic geometry.        
         funcTime(
             calcNearestAndFurthestDistances_Parallel, pointsRandom[0], pointsRandom[1], basicGeo
         );

    // // USING FASTER ALGO
    //     funcTime(
    //         calcNearestAndFurthestDistances_Serial_Fast, pointsRandom[0], pointsRandom[1], basicGeo
    //     );

    //     funcTime(
    //         calcNearestAndFurthestDistances_Parallel_Fast, pointsRandom[0], pointsRandom[1], basicGeo
    //     );    
    
    // // USING GRID METHOD
    //     funcTime(
    //         calculateNearestsWithGridMethod, pointsRandom[0], pointsRandom[1]
    //     );

    return 0;
}