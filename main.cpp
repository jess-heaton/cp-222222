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

// This function checks all points and figures out how close or far each one is to others.
// It's done without recalculating distances for pairs more than once to make it quicker.
// Runs in serial mode (not parallelized).
void computeDistancesSequentially_efficient(
    vector<double> xCoords, vector<double> yCoords, void (*distanceLogic)(double, double, double[])
) {
    // Figure out how many points we have to work with
    int totalPoints = xCoords.size();

    // Make sure the x and y lists have the same size
    if (totalPoints != yCoords.size()) {
        throw invalid_argument("The x and y coordinate arrays are mismatched. Check input sizes!");
    }

    // Prepare arrays to keep the smallest and largest distances for each point
    vector<double> shortestDistances(totalPoints, 1000.0); // Use a large starting number
    vector<double> largestDistances(totalPoints, -1000.0); // Use a small starting number

    // Temporary array to store results of distance calculations
    double distanceResults[2]; // One for nearest, one for farthest

    // Loop through every point
    for (int currentPoint = 0; currentPoint < totalPoints; ++currentPoint) {
        // Start comparing this point with others
        for (int comparePoint = currentPoint + 1; comparePoint < totalPoints; ++comparePoint) {
            // Get the difference in x and y
            double xDiff = abs(xCoords[comparePoint] - xCoords[currentPoint]);
            double yDiff = abs(yCoords[comparePoint] - yCoords[currentPoint]);

            // Call the geometry function to calculate distances
            distanceLogic(xDiff, yDiff, distanceResults);

            // Update the nearest and furthest distances for the current point
            if (distanceResults[0] < shortestDistances[currentPoint]) {
                shortestDistances[currentPoint] = distanceResults[0];
            }
            if (distanceResults[1] > largestDistances[currentPoint]) {
                largestDistances[currentPoint] = distanceResults[1];
            }

            // Update the nearest and furthest distances for the other point (comparePoint)
            if (distanceResults[0] < shortestDistances[comparePoint]) {
                shortestDistances[comparePoint] = distanceResults[0];
            }
            if (distanceResults[1] > largestDistances[comparePoint]) {
                largestDistances[comparePoint] = distanceResults[1];
            }
        }
    }

    // Compute the averages for all nearest and furthest distances
    double sumOfNearest = 0.0;
    double sumOfFurthest = 0.0;

    for (int i = 0; i < totalPoints; ++i) {
        // Take square roots of distances since they were squared earlier
        shortestDistances[i] = sqrt(shortestDistances[i]);
        largestDistances[i] = sqrt(largestDistances[i]);

        // Add to the totals
        sumOfNearest += shortestDistances[i];
        sumOfFurthest += largestDistances[i];
    }

    double averageNearest = sumOfNearest / totalPoints;
    double averageFurthest = sumOfFurthest / totalPoints;

    // Output the results
    outputRes(shortestDistances, largestDistances, averageNearest, averageFurthest, false, distanceLogic == calculateWraparoundDistances);
}


// Function to calculate the smallest and largest distances for points using parallelism.
// Distances are calculated based on the provided geometry calculation function.
// This function avoids recalculating distances twice to improve speed.
void computeDistancesWithParallelization_efficient(
    vector<double> xCoords, vector<double> yCoords, void (*distanceCalculator)(double, double, double[])
) {
    // Get the number of points
    int pointCount = xCoords.size();

    // Ensure x and y have the same number of points
    if (pointCount != yCoords.size()) {
        throw invalid_argument("Mismatch between x and y coordinate sizes. Ensure they match!");
    }

    // Prepare vectors to store the nearest and furthest distances
    vector<double> smallestDistances(pointCount, 99999.9); // Start with a big value for "smallest"
    vector<double> largestDistances(pointCount, -99999.9); // Start with a tiny value for "largest"

    // Parallel region
    #pragma omp parallel
    {
        // Temporary storage for thread-local results
        double localResults[2];

        // Loop over all points in parallel
        #pragma omp for schedule(dynamic, 32) // Chunk size of 32
        for (int currentIdx = 0; currentIdx < pointCount; ++currentIdx) {
            // Temporary variables to track nearest and furthest for the current point
            double localSmallestSquared = 99999.9;
            double localLargestSquared = -99999.9;

            // Compare this point with all points after it
            for (int compareIdx = currentIdx + 1; compareIdx < pointCount; ++compareIdx) {
                // Calculate differences in x and y
                double xDiff = abs(xCoords[compareIdx] - xCoords[currentIdx]);
                double yDiff = abs(yCoords[compareIdx] - yCoords[currentIdx]);

                // Call the provided function to calculate distances
                distanceCalculator(xDiff, yDiff, localResults);

                // Update the nearest and furthest distances for the current point
                if (localResults[0] < localSmallestSquared) {
                    localSmallestSquared = localResults[0];
                }
                if (localResults[1] > localLargestSquared) {
                    localLargestSquared = localResults[1];
                }

                // Update the nearest and furthest distances for the comparison point
                #pragma omp critical
                {
                    if (localResults[0] < smallestDistances[compareIdx]) {
                        smallestDistances[compareIdx] = localResults[0];
                    }
                    if (localResults[1] > largestDistances[compareIdx]) {
                        largestDistances[compareIdx] = localResults[1];
                    }
                }
            }

            // Save the results for the current point
            #pragma omp critical
            {
                if (localSmallestSquared < smallestDistances[currentIdx]) {
                    smallestDistances[currentIdx] = localSmallestSquared;
                }
                if (localLargestSquared > largestDistances[currentIdx]) {
                    largestDistances[currentIdx] = localLargestSquared;
                }
            }
        }
    }

    // Variables to calculate averages
    double totalNearest = 0.0;
    double totalFurthest = 0.0;

    // Compute square roots and accumulate sums
    #pragma omp parallel for reduction(+:totalNearest, totalFurthest)
    for (int idx = 0; idx < pointCount; ++idx) {
        smallestDistances[idx] = sqrt(smallestDistances[idx]);
        largestDistances[idx] = sqrt(largestDistances[idx]);

        totalNearest += smallestDistances[idx];
        totalFurthest += largestDistances[idx];
    }

    // Calculate the averages
    double averageNearest = totalNearest / pointCount;
    double averageFurthest = totalFurthest / pointCount;

    // Output the results
    outputRes(
        smallestDistances, 
        largestDistances, 
        averageNearest, 
        averageFurthest, 
        true, // Indicates parallel execution
        distanceCalculator == calculateWraparoundDistances // Whether wraparound geometry was used
    );
}




int main() {
    // Set the number of threads for OpenMP to use

    // omp_set_num_threads(4);
    // omp_set_num_threads(8);
    omp_set_num_threads(10);

    // Configure output format for floating-point numbers
    cout << std::fixed; // Fixed-point notation
    cout << std::setprecision(10); // Display up to 10 decimal places

    // Define the number of points to generate
    int numberOfPoints = 100000;

    // Generate a set of random points for testing
    vector<vector<double>> randomGeneratedPoints = generateRandomPoints(numberOfPoints);

    // Read points from the CSV file
    vector<vector<double>> pointsFromCsvFile;
    utils::easyio::readCsv("100000 locations.csv", pointsFromCsvFile);

    // Test 1: Standard Naive Serial
    cout << "Standard Naive Serial..." << endl;
    measureExecutionTime(
        computeDistancesSequentially,
        randomGeneratedPoints[0],
        randomGeneratedPoints[1],
        calculateEucDistance
    );

    // Test 2: Wraparound Naive Serial
    cout << "Wraparound Naive Serial..." << endl;
    measureExecutionTime(
        computeDistancesSequentially,
        randomGeneratedPoints[0],
        randomGeneratedPoints[1],
        calculateWraparoundDistances
    );

    // Test 3: Standard Naive Parallel
    cout << "Standard Naive Parallel..." << endl;
    measureExecutionTime(
        computeDistancesWithParallelization,
        randomGeneratedPoints[0],
        randomGeneratedPoints[1],
        calculateEucDistance
    );

    // Test 4: Wraparound Naive Parallel
    cout << "Wraparound Naive Parallel..." << endl;
    measureExecutionTime(
        computeDistancesWithParallelization,
        randomGeneratedPoints[0],
        randomGeneratedPoints[1],
        calculateWraparoundDistances
    );

    // Test 5: Standard Fast Serial
    cout << "Standard Fast Serial..." << endl;
    measureExecutionTime(
        computeDistancesSequentially_efficient,
        randomGeneratedPoints[0],
        randomGeneratedPoints[1],
        calculateEucDistance
    );

    // Test 6: Wraparound Fast Serial
    cout << "Wraparound Fast Serial..." << endl;
    measureExecutionTime(
        computeDistancesSequentially_efficient,
        randomGeneratedPoints[0],
        randomGeneratedPoints[1],
        calculateWraparoundDistances
    );

    // Test 7: Standard Fast Parallel
    cout << "Standard Fast Parallel..." << endl;
    measureExecutionTime(
        computeDistancesWithParallelization_efficient,
        randomGeneratedPoints[0],
        randomGeneratedPoints[1],
        calculateEucDistance
    );

    // Test 8: Wraparound Fast Parallel
    cout << "Wraparound Fast Parallel..." << endl;
    measureExecutionTime(
        computeDistancesWithParallelization_efficient,
        randomGeneratedPoints[0],
        randomGeneratedPoints[1],
        calculateWraparoundDistances
    );

    return 0;
}
