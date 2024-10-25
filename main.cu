#include "FlyingEdgesAlgorithm.h"

// Function to read scalar data from file
template<typename T>
std::vector<T> readF32File(const std::string &filename, std::size_t numElements) {
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // Get file size
    file.seekg(0, std::ios::end);
    std::size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Calculate the number of elements to read
    std::size_t dataSize = std::min(fileSize / sizeof(T), numElements);
    std::vector<T> data(dataSize);

    // Read data
    file.read(reinterpret_cast<char *>(data.data()), dataSize * sizeof(T));

    if (!file) {
        throw std::runtime_error("Error reading file: " + filename);
    }

    return data;
}

int main() {
    try {
        float isovalue = 8453;
        std::string filePath = "/home/exouser/dataset/SDRBENCH-EXASKY-NYX-512x512x512/temperature.f32";
        size_t numElements = 512 * 512 * 512;
        dim3 dataShape(512, 512, 512);
        std::vector<float> host_scalars;

        std::cout << "Reading file..." << std::endl;
        try {
            host_scalars = readF32File<float>(filePath, numElements);
        } catch (const std::exception &e) {
            std::cerr << "Error reading file: " << e.what() << std::endl;
            return -1;
        }
        std::cout << "File read successfully. Elements: " << host_scalars.size() << std::endl;

        // Create an instance of FlyingEdgesAlgorithm
        FlyingEdgesAlgorithm flyingEdges(host_scalars.data(), isovalue, dataShape);

        // Execute the algorithm
        flyingEdges.execute();

        // Save results to OBJ file
        // flyingEdges.saveResultsToOBJ("output.obj");
        std::cout << "Done" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return -1;
    }
}
