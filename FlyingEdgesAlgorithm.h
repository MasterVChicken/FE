//
// Created by Yanliang Li on 10/25/24.
//

#ifndef FLYING_EDGES_ALGORITHM_H
#define FLYING_EDGES_ALGORITHM_H

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <stdio.h>
#include <cub/device/device_scan.cuh>

// Helper macro for checking CUDA errors
#define CHECK_CUDA(func) checkCudaError((func), #func, __FILE__, __LINE__)

// Helper function to check CUDA errors
inline void checkCudaError(cudaError_t result, const char *func, const char *file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result)
                  << " (" << cudaGetErrorString(result) << ") \"" << func << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Vertex and Triangle structures
struct Vertex {
    float x, y, z;
};

struct Triangle {
    int v1, v2, v3;
};

// __constant__ unsigned char numTris[256];
// __constant__ bool isCut[256][12];
// __constant__ char caseTriangles[256][16];
// __constant__ unsigned int edgeVertices[12][2];

template<typename T>
__global__ void calculateEdgeCases(T *scalars, T isovalue, int *edgeCases, dim3 dataShape);

__global__ void calculateTrimPositions(int *edgeCases, int *leftTrim, int *rightTrim, dim3 dataShape);

__global__ void getCubeCases(const int *edgeCases, const int *leftTrim, const int *rightTrim, int *triCount,
                             int *cubeCases, int *left_c, int *right_c, dim3 dataShape);

__global__ void getCubeTris(int *cubeCases, int *tri_nums, dim3 dataShape);

template<typename T>
__global__ void Pass4Kernel(const T *scalars, const int *cubeCases, const int *triOffsets,
                            Vertex *vertices, Triangle *triangles, T isovalue, dim3 dataShape);

// FlyingEdgesAlgorithm class
class FlyingEdgesAlgorithm {
public:
    FlyingEdgesAlgorithm(float *h_scalars, float isovalue, dim3 dataShape);
    ~FlyingEdgesAlgorithm();

    void execute();
    void saveResultsToOBJ(const std::string &filename);

    template<typename T>
    static __device__ int calculateEdgeCase(T left, T right, T isovalue);

    static __device__ int getCubeCase(int ec0, int ec1, int ec2, int ec3);

    template<typename T>
    static __device__ void interpolateVertex(float *vert, T isovalue,
                                             float x1, float y1, float z1, T val1,
                                             float x2, float y2, float z2, T val2);

private:
    // Member variables
    float *d_scalars;
    int *d_edgeCases;
    int *d_cubeCases;
    int *d_leftTrim;
    int *d_rightTrim;
    int *d_leftTrim_c;
    int *d_rightTrim_c;
    int *d_triCount;
    int *d_triOffsets;
    Vertex *d_vertices;
    Triangle *d_triangles;
    float isovalue;
    dim3 dataShape;
    int totalVoxels;
    int totalTriangles;
    int totalVertices;



    // Member functions
    template<typename T>
    void Pass1(T *scalars, T isovalue, int *edgeCases, int *leftTrim, int *rightTrim, dim3 dataShape);

    void Pass2(int *d_edgeCases, int *d_leftTrim, int *d_rightTrim, int *d_triCount,
               int *d_cubeCases, int *d_left_c, int *d_right_c, dim3 dataShape);

    void Pass3(int *d_cubeCases, int *d_triOffsets, dim3 dataShape);

    template<typename T>
    void Pass4(const T *d_scalars, const int *d_cubeCases, const int *d_triOffsets,
               Vertex *d_vertices, Triangle *d_triangles, T isovalue, dim3 dataShape);
};

#endif // FLYING_EDGES_ALGORITHM_H
