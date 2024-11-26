#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <utility>
#include <omp.h>
#include <iomanip> 
#include "easyio.h"
#include "rng.h"

using namespace std; 

// Generates a list of random points (x, y) between 0 and 1
vector<vector<double>> generateRandomPoints(int totalPoints) {
    // Define the range for random values
    double minVal = 0;
    double maxVal = 1;

    // Create vectors to store x and y coordinates
    vector<double> xCoords(totalPoints, 0);
    vector<double> yCoords(totalPoints, 0);

    // Loop to fill x and y with random numbers
    for (int i = 0; i < totalPoints; i++) {
        // Generate random numbers for x and y
        xCoords[i] = minVal + (maxVal - minVal) * ((double)rand() / RAND_MAX);
        yCoords[i] = minVal + (maxVal - minVal) * ((double)rand() / RAND_MAX);
    }

    // Combine x and y into a 2D vector and return it
    vector<vector<double>> randomPoints = {xCoords, yCoords};
    return randomPoints;
}

void measureExecutionTime(void (*func)(vector<double>, vector<double>, void (*)(double, double, double[])),
                          vector<double> x, vector<double> y, void (*calcFunc)(double, double, double[])) {
    // Start time
    auto start = std::chrono::high_resolution_clock::now();

    // Call the provided function
    func(x, y, calcFunc);

    // End time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration in seconds
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    // Print the result
    cout << "Execution time: " << elapsed << " seconds\n";
}

void outputRes(
    vector<double> &nearests, vector<double> &furthests, double avgNearest, double avgFurthest,
    bool isParallel, bool isWraparoundGeo
) {
    // Write nearest distances to a file
    ofstream nearestFile("basicNearests.txt");
    for (size_t i = 0; i < nearests.size(); ++i) {
        nearestFile << nearests[i] << endl;
    }
    nearestFile.close(); // Close the file manually

    // Write furthest distances to another file
    ofstream furthestFile("basicFurthests.txt");
    for (size_t i = 0; i < furthests.size(); ++i) {
        furthestFile << furthests[i] << endl;
    }
    furthestFile.close(); // Close the file manually

    // Print the title based on parallel/serial and geometry type
    cout << "Calculating in ";
    if (isParallel) {
        cout << "parallel ";
    } else {
        cout << "serial ";
    }
    cout << "using ";
    if (isWraparoundGeo) {
        cout << "wraparound geometry" << endl;
    } else {
        cout << "basic geometry" << endl;
    }

    // Print the averages to the console
    cout << "Mean nearest distance: " << avgNearest << endl;
    cout << "Mean furthest distance: " << avgFurthest << endl;
}

// Function to calculate the squared distance between two points
// This is used for basic geometry where no wraparound is applied
void calculateEucDistance(double differenceInX, double differenceInY, double outputArray[]) {
    // Step 1: Calculate the square of the x difference
    double squareOfXDifference = differenceInX * differenceInX;

    // Step 2: Calculate the square of the y difference
    double squareOfYDifference = differenceInY * differenceInY;

    // Step 3: Add the squared differences to find the squared distance
    double totalSquaredDistance = squareOfXDifference + squareOfYDifference;

    // Step 4: Save the result to both positions in the output array
    outputArray[0] = totalSquaredDistance; // Save to first slot
    outputArray[1] = totalSquaredDistance; // Save to second slot (same value for both)
}


// Function to calculate the shortest and longest distances with wraparound geometry
// Wraparound means distances are measured across edges of a toroidal space
void calculateWraparoundDistances(double differenceInX, double differenceInY, double outputArray[]) {
    // Variables to hold the shortest and longest distances
    double smallestDistanceSquared = 0.0;
    double largestDistanceSquared = 0.0;

    // Step 1: Calculate the "wrapped" difference in Y direction
    double wrappedYDifference1 = differenceInY; // Original Y difference
    double wrappedYDifference2 = 1 - differenceInY; // Wraparound Y difference
    double minimumYDifference = wrappedYDifference1; // Start with original as minimum

    if (wrappedYDifference2 < wrappedYDifference1) {
        // If wrapped difference is smaller, update the minimum
        minimumYDifference = wrappedYDifference2;
    }

    // Step 2: Calculate the "wrapped" difference in X direction
    double wrappedXDifference1 = differenceInX; // Original X difference
    double wrappedXDifference2 = 1 - differenceInX; // Wraparound X difference
    double minimumXDifference = wrappedXDifference1; // Start with original as minimum

    if (wrappedXDifference2 < wrappedXDifference1) {
        // If wrapped difference is smaller, update the minimum
        minimumXDifference = wrappedXDifference2;
    }

    // Step 3: Compute the shortest distance (squared)
    smallestDistanceSquared = (minimumYDifference * minimumYDifference) +
                              (minimumXDifference * minimumXDifference);

    // Step 4: Now calculate the "wrapped" difference for the furthest distance
    double maximumYDifference = wrappedYDifference1; // Start with original as maximum
    if (wrappedYDifference2 > wrappedYDifference1) {
        // If wrapped difference is larger, update the maximum
        maximumYDifference = wrappedYDifference2;
    }

    double maximumXDifference = wrappedXDifference1; // Start with original as maximum
    if (wrappedXDifference2 > wrappedXDifference1) {
        // If wrapped difference is larger, update the maximum
        maximumXDifference = wrappedXDifference2;
    }

    // Step 5: Compute the furthest distance (squared)
    largestDistanceSquared = (maximumYDifference * maximumYDifference) +
                             (maximumXDifference * maximumXDifference);

    // Step 6: Save results to the output array
    outputArray[0] = smallestDistanceSquared; // Save shortest distance squared
    outputArray[1] = largestDistanceSquared;  // Save longest distance squared
}






// For every point, figure out how close and far it is to/from the other points.
// This is done in a plain straightforward way without parallelization.
// Outputs results to files and prints the average distances.
// Works by accepting a geometry calculator function (e.g., for basic or wraparound geometry).
void computeDistancesSequentially(
    vector<double> xCoordinates, vector<double> yCoordinates, void (*geometryCalculator)(double, double, double[])
) {
    // Number of points
    int totalPoints = xCoordinates.size();

    // Check if sizes of x and y are equal
    if (totalPoints != yCoordinates.size()) {
        throw invalid_argument("The x and y coordinates must have the same number of points.");
    }

    // Prepare storage for nearest and furthest distances for each point
    vector<double> smallestDistances(totalPoints, 0);
    vector<double> largestDistances(totalPoints, 0);

    // Temporary array to hold calculated distances
    double distanceResults[2];

    // Loop through all the points
    for (int currentPoint = 0; currentPoint < totalPoints; ++currentPoint) {
        // Use very large and small numbers to initialize nearest and furthest trackers
        double nearestSoFarSquared = 99999.9; // Arbitrary large number
        double furthestSoFarSquared = -99999.9; // Arbitrary small number

        // Compare this point with all other points
        for (int comparisonPoint = 0; comparisonPoint < totalPoints; ++comparisonPoint) {
            // Skip the current point itself
            if (currentPoint == comparisonPoint) continue;

            // Compute differences in x and y
            double xDifference = abs(xCoordinates[comparisonPoint] - xCoordinates[currentPoint]);
            double yDifference = abs(yCoordinates[comparisonPoint] - yCoordinates[currentPoint]);

            // Use the geometry calculator to compute distances
            geometryCalculator(xDifference, yDifference, distanceResults);

            // Check if this distance is the smallest we've seen so far
            if (distanceResults[0] < nearestSoFarSquared) {
                nearestSoFarSquared = distanceResults[0];
            }

            // Check if this distance is the largest we've seen so far
            if (distanceResults[1] > furthestSoFarSquared) {
                furthestSoFarSquared = distanceResults[1];
            }
        }

        // Save the results for this point after taking square roots
        smallestDistances[currentPoint] = sqrt(nearestSoFarSquared);
        largestDistances[currentPoint] = sqrt(furthestSoFarSquared);
    }

    // Calculate averages for all nearest and furthest distances
    double totalNearestDistance = 0;
    double totalFurthestDistance = 0;
    for (int i = 0; i < totalPoints; ++i) {
        totalNearestDistance += smallestDistances[i];
        totalFurthestDistance += largestDistances[i];
    }
    double averageNearest = totalNearestDistance / totalPoints;
    double averageFurthest = totalFurthestDistance / totalPoints;

    // Output the results
    outputRes(smallestDistances, largestDistances, averageNearest, averageFurthest, false, geometryCalculator == calculateWraparoundDistances);
}
// Parallel computation of nearest and furthest distances for each point in a set.
// This function uses OpenMP for parallelization.
// The "geometryLogicFunction" determines the calculation logic (e.g., basic or wraparound).
void computeDistancesWithParallelization(
    vector<double> xCoordinates, vector<double> yCoordinates, void (*geometryLogicFunction)(double, double, double[])
) {
    // Step 1: Verify that the input vectors have matching sizes
    int numberOfPoints = xCoordinates.size();
    if (numberOfPoints != yCoordinates.size()) {
        throw invalid_argument("The x and y coordinate arrays must have the same number of elements.");
    }

    // Step 2: Prepare storage for results
    vector<double> nearestDistances(numberOfPoints, 0.0);  // To hold nearest distances
    vector<double> furthestDistances(numberOfPoints, 0.0); // To hold furthest distances

    // Step 3: Set up a reusable array for results per pair of points
    double tempResults[2]; // Holds {nearest distance squared, furthest distance squared}

    // Step 4: Loop over all points in parallel
    #pragma omp parallel for private(tempResults)
    for (int currentIndex = 0; currentIndex < numberOfPoints; ++currentIndex) {
        // Initialize running minimum and maximum distances for the current point
        double smallestSquaredDistance = 2.1; // Arbitrary large enough value
        double largestSquaredDistance = -0.1; // Arbitrary small enough value

        // Step 4a: Iterate over all other points to calculate distances
        for (int comparisonIndex = 0; comparisonIndex < numberOfPoints; ++comparisonIndex) {
            // Skip comparing a point with itself
            if (currentIndex == comparisonIndex) {
                continue;
            }

            // Calculate the absolute differences in x and y coordinates
            double xDifference = abs(xCoordinates[comparisonIndex] - xCoordinates[currentIndex]);
            double yDifference = abs(yCoordinates[comparisonIndex] - yCoordinates[currentIndex]);

            // Call the geometry logic function to calculate distances
            geometryLogicFunction(xDifference, yDifference, tempResults);

            // Update the running nearest and furthest distances
            if (tempResults[0] < smallestSquaredDistance) {
                smallestSquaredDistance = tempResults[0];
            }
            if (tempResults[1] > largestSquaredDistance) {
                largestSquaredDistance = tempResults[1];
            }
        }

        // Store the results for the current point
        nearestDistances[currentIndex] = sqrt(smallestSquaredDistance);
        furthestDistances[currentIndex] = sqrt(largestSquaredDistance);
    }

    // Step 5: Compute the averages of the nearest and furthest distances
    double totalNearestSum = 0.0;
    double totalFurthestSum = 0.0;

    #pragma omp parallel for reduction(+:totalNearestSum, totalFurthestSum)
    for (int pointIndex = 0; pointIndex < numberOfPoints; ++pointIndex) {
        totalNearestSum += nearestDistances[pointIndex];
        totalFurthestSum += furthestDistances[pointIndex];
    }

    double averageNearestDistance = totalNearestSum / numberOfPoints;
    double averageFurthestDistance = totalFurthestSum / numberOfPoints;

    // Step 6: Output the results using the provided function
    outputRes(
        nearestDistances, 
        furthestDistances, 
        averageNearestDistance, 
        averageFurthestDistance, 
        true, // Indicates parallel computation
        geometryLogicFunction == calculateWraparoundDistances // Flag for geometry type
    );
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
    outputRes(nearests, furthests, avgNearest, avgFurthest, false, calcFunc == calculateWraparoundDistances);
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
    outputRes(nearests, furthests, avgNearest, avgFurthest, true, calcFunc == calculateWraparoundDistances);
}

int main () {
    omp_set_num_threads(10);
    cout << fixed << setprecision(10); // print doubles with 10 decimal places
    
    int n = 100000;
    
    vector<vector<double>> pointsRandom = generateRandomPoints(n);

    vector<vector<double>> pointsCSV;
    utils::easyio::readCsv("100000 locations.csv", pointsCSV);

    // examples
    
    // // SERIAL
    // // n randomly-initialised points, using basic geometry.
    //     measureExecutionTime(
      //       calcNearestAndFurthestDistances_Serial, pointsRandom[0], pointsRandom[1], calculateEucDistance
    //     );
    // // n randomly-initialised points, using wraparound geometry.
    //     measureExecutionTime(
    //         calcNearestAndFurthestDistances_Serial, pointsRandom[0], pointsRandom[1], calculateWraparoundDistances
    //     );
    // // n csv-initialised points, using basic geometry.
    //     measureExecutionTime(
    //         calcNearestAndFurthestDistances_Serial, pointsCSV[0], pointsCSV[1], calculateEucDistance
    //     );

    // // PARALLELISATION
    // // n randomly-initialised points, using basic geometry.        
         measureExecutionTime(
             computeDistancesWithParallelization, pointsRandom[0], pointsRandom[1], calculateEucDistance
         );

    // // USING FASTER ALGO
    //     measureExecutionTime(
    //         calcNearestAndFurthestDistances_Serial_Fast, pointsRandom[0], pointsRandom[1], calculateEucDistance
    //     );

    //     measureExecutionTime(
    //         calcNearestAndFurthestDistances_Parallel_Fast, pointsRandom[0], pointsRandom[1], calculateEucDistance
    //     );    
    
    return 0;
}