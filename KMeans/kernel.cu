
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <curand.h>

#include "KMeans_cpu.h"
#include "helper_cuda.h"


using namespace std;
using namespace cv;

const int MAX_ITERS = 100;

__global__ void clustersUpdateGPU(
    uint8_t* image,
    uint8_t* clusters,
    int* centroids,
    bool* changed,
    int rows, 
    int cols,
    int K
){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    if (idx >= rows * cols) {
        return;
    }

    extern __shared__ int s_centroids[];

    int val;
    if (tidx < K) {
        int v1, v2, v3;
        v1 = centroids[3 * tidx];
        v2 = centroids[3 * tidx + 1];
        v3 = centroids[3 * tidx + 2];
        s_centroids[3 * tidx] = v1;
        s_centroids[3 * tidx + 1] = v2;
        s_centroids[3 * tidx + 2] = v3;
        s_centroids[3 * K + tidx] = v1 * v1 + v2 * v2 + v3 * v3;
    }

    __syncthreads();

    uint8_t act_best;
    int x1, x2, x3;
    int act_dist, new_dist;
    act_best = clusters[idx];
    x1 = image[3 * idx];
    x2 = image[3 * idx + 1];
    x3 = image[3 * idx + 2];
    act_dist = -2 * x1 * s_centroids[3 * act_best] - 2 * x2 * s_centroids[3 * act_best + 1] - 2 * x3 * s_centroids[3 * act_best + 2] + s_centroids[3 * K + act_best];

    for (int j = 0; j < K; j++) {
        new_dist = -2 * x1 * s_centroids[3 * j] - 2 * x2 * s_centroids[3 * j + 1] - 2 * x3 * s_centroids[3 * j + 2] + s_centroids[3 * K + j];
        if (new_dist < act_dist) {
            changed[0] = true;
            act_best = (uint8_t)j;
            act_dist = new_dist;
        }
    }
    clusters[idx] = act_best;
}

// Dist calculations without shared memory:

__device__ int dist_gpu(uint8_t* image, int* centroids, int i, int j) {
    int val;
    int dist = 0;
    for (int k = 0; k < 3; k++) {
        val = (int)image[3 * i + k] - centroids[3 * j + k];
        dist += val * val;
    }
    return dist;
}

__global__ void clustersUpdateGPU_WS(
    uint8_t* image,
    uint8_t* clusters,
    int* centroids,
    bool* changed,
    int rows,
    int cols,
    int K
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) {
        return;
    }

    uint8_t act_best;
    int act_dist, new_dist;
    act_best = clusters[idx];
    act_dist = dist_gpu(image, centroids, idx, act_best);

    for (int j = 0; j < K; j++) {
        new_dist = dist_gpu(image, centroids, idx, j);
        if (new_dist < act_dist) {
            changed[0] = true;
            act_best = (uint8_t)j;
            act_dist = new_dist;
        }
    }
    clusters[idx] = act_best;
}

__global__ void centroidsSumGPU(
    uint8_t* image,
    uint8_t* clusters,
    int* centroids,
    int* clusters_sizes,
    int rows,
    int cols,
    int K
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_ = blockIdx.x * blockDim.x;
    int tidx = threadIdx.x;
    if (idx >= rows * cols) {
        return;
    }

    extern __shared__ int centroids_info[];

    if (tidx >= 4) {
        return;
    }

    for (int k = 0; k < K; k++) {
        centroids_info[4 * k + tidx] = 0;
    }

    __syncthreads();

    for (int i = 0; i < blockDim.x; i++) {
        if (idx_ + i >= rows * cols) {
            break;
        }
        int cluster = (int)clusters[idx_ + i];
        if (tidx < 3) {
            centroids_info[4 * cluster + tidx] += image[(idx_ + i) * 3 + tidx];
        }
        else {
            centroids_info[4 * cluster + 3]++;
        }
    }

    __syncthreads();

    for (int k = 0; k < K; k++) {
        if (tidx < 3) {
            atomicAdd(&centroids[3 * k + tidx], centroids_info[4 * k + tidx]);
        }
        else {
            atomicAdd(&clusters_sizes[k], centroids_info[4 * k + 3]);
        }
    }

}

__global__ void centroidsMeanGPU(
    int* centroids,
    int* clusters_sizes,
    float* rand,
    int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) {
        return;
    }

    int cluster_size = clusters_sizes[idx];
    //printf("Idx: %d, size: %d, color (%d,%d,%d)\n", idx, cluster_size, centroids[3 * idx], centroids[3 * idx + 1], centroids[3 * idx + 2]);
    if (cluster_size > 0) {
        centroids[3*idx] /= cluster_size;
        centroids[3*idx + 1] /= cluster_size;
        centroids[3*idx + 2] /= cluster_size;
    }
    else {
        centroids[3 * idx] = 255.0 * rand[3*idx];
        centroids[3 * idx + 1] = 255.0 * rand[3 * idx + 1];
        centroids[3 * idx + 2] = 255.0 * rand[3 * idx + 2];
    }
    //printf("Idx: %d, size: %d, color (%d,%d,%d)\n", idx, cluster_size, centroids[3*idx], centroids[3*idx + 1], centroids[3*idx + 2]);
}

__global__ void printGPU(
    uint8_t* image,
    uint8_t* clusters,
    int* centroids,
    int* clusters_sizes,
    int rows,
    int cols,
    int K
) {
    printf("GPU:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", (int)clusters[i * cols + j]);
        }
        printf("\n");
    }

    for (int i = 0; i < K; i++) {
        printf("k: %d, color: (%d,%d,%d)\n", i, centroids[3 * i], centroids[3 * i + 1], centroids[3 * i + 2]);
    }
}

void kMeansClusteringGPU(
    uint8_t* h_image,
    uint8_t* h_reduced_image_gpu,
    uint8_t* h_clusters_gpu,
    int* h_centroids_gpu,
    int* h_clusters_sizes_gpu,
    bool* h_changed,
    uint8_t* d_image,
    uint8_t* d_clusters_gpu,
    int* d_centroids_gpu,
    int* d_clusters_size_gpu,
    bool* d_changed,
    float* d_rand,
    curandGenerator_t* gen,
    int rows,
    int cols,
    int K
){
    const int THREADS = max(1024, K);
    const int THREADS2 = 512;
    for (int t = 0; t < MAX_ITERS; t++) {
        h_changed[0] = false;
        checkCudaErrors(cudaMemcpy(d_changed, h_changed, sizeof(bool), cudaMemcpyHostToDevice));

        clustersUpdateGPU<<<(rows * cols + THREADS - 1) / THREADS, THREADS, 4 * K * sizeof(int)>>>(
        //clustersUpdateGPU_WS<<<(rows * cols + THREADS - 1) / THREADS, THREADS>>>(
            d_image,
            d_clusters_gpu,
            d_centroids_gpu,
            d_changed,
            rows,
            cols,
            K
            );
        getLastCudaError("clusters update kernel execution failed");
        cudaDeviceSynchronize();
        getLastCudaError("clusters update kernel execution failed after synchronize");

        checkCudaErrors(cudaMemcpy(h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
        if (t > 0 && !h_changed[0]) {
            break;
        }

        checkCudaErrors(cudaMemset(d_centroids_gpu, 0, K * 3 * sizeof(int)));
        checkCudaErrors(cudaMemset(d_clusters_size_gpu, 0, K * sizeof(int)));

        centroidsSumGPU<<<(rows * cols + THREADS2 - 1) / THREADS2, THREADS2, 4 * K * sizeof(int)>>>(
            d_image,
            d_clusters_gpu,
            d_centroids_gpu,
            d_clusters_size_gpu,
            rows,
            cols,
            K
            );
        getLastCudaError("centroids sum kernel execution failed");
        cudaDeviceSynchronize();
        getLastCudaError("centroids sum kernel execution failed after synchronize");

        curandGenerateUniform(*gen, d_rand, 3 * K);

        centroidsMeanGPU<<<1, K>>>(
            d_centroids_gpu,
            d_clusters_size_gpu,
            d_rand,
            K
            );
        getLastCudaError("centroids mean kernel execution failed");
        cudaDeviceSynchronize();
        getLastCudaError("centroids mean kernel execution failed after synchronize");
        //printGPU<<<1,1>>>(d_image,d_clusters_gpu,d_centroids_gpu,d_clusters_size_gpu,rows,cols,K);
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(h_clusters_gpu, d_clusters_gpu, rows * cols * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_centroids_gpu, d_centroids_gpu, K * 3 * sizeof(int), cudaMemcpyDeviceToHost));
}

int main(int argc, char* argv[])
{
    // -----------------------------------------------------------------------------------------
    // Input parsing

    if (argc < 3) {
        cout << "Invalid input!" << endl;
        return 0;
    }
    const string PATH = argv[1];
    long arg = strtol(argv[2], NULL, 10);
    const uint8_t K = arg;

    srand(time(0));

    // -----------------------------------------------------------------------------------------
    // Image reading

    Mat image = imread(PATH);
    if (image.empty())
    {
        cout << "Could not open or find the image" << endl;
        cin.get();
        return -1;
    }

    int rows, cols;
    rows = image.rows;
    cols = image.cols;

    // -----------------------------------------------------------------------------------------
    // Memory allocation

    uint8_t *h_image = (uint8_t*)malloc(rows * cols * 3 * sizeof(uint8_t));
    uint8_t* h_reduced_image_cpu = (uint8_t*)malloc(rows * cols * 3 * sizeof(uint8_t));
    uint8_t* h_reduced_image_gpu = (uint8_t*)malloc(rows * cols * 3 * sizeof(uint8_t));
    memcpy(h_image, image.data, rows * cols * 3 * sizeof(uint8_t));

    uint8_t *h_clusters_cpu = (uint8_t*)malloc(rows * cols * sizeof(uint8_t));
    uint8_t* h_clusters_gpu = (uint8_t*)malloc(rows * cols * sizeof(uint8_t));

    int *h_centroids_cpu = (int*)malloc(K * 3 * sizeof(int));
    int *h_centroids_gpu = (int*)malloc(K * 3 * sizeof(int));

    int* h_clusters_sizes_cpu = (int*)malloc(K * sizeof(int));
    int* h_clusters_sizes_gpu = (int*)malloc(K * sizeof(int));

    bool* h_changed = (bool*)malloc(sizeof(bool));

    uint8_t* d_image;
    uint8_t* d_clusters_gpu;
    int* d_centroids_gpu;
    int* d_clusters_size_gpu;
    bool* d_changed;
    float* d_rand;

    cudaMalloc(&d_image, rows * cols * 3 * sizeof(uint8_t));
    cudaMalloc(&d_clusters_gpu, rows * cols * sizeof(uint8_t));
    cudaMalloc(&d_centroids_gpu, K * 3 * sizeof(int));
    cudaMalloc(&d_clusters_size_gpu, K * sizeof(int));
    cudaMalloc(&d_changed, sizeof(bool));
    cudaMalloc(&d_rand, K * 3 * sizeof(float));

    // -----------------------------------------------------------------------------------------
    // Data initialization

    for (int i = 0; i < 3 * K; i++) {
        h_centroids_cpu[i] = rand() % 256;
    }
    for (int i = 0; i < rows * cols; i++) {
        h_clusters_cpu[i] = (uint8_t)0;
        h_clusters_gpu[i] = (uint8_t)0;
    }

    memcpy(h_centroids_gpu, h_centroids_cpu, K * 3 * sizeof(int));

    cudaMemcpy(d_image, h_image, rows * cols * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids_gpu, h_centroids_gpu, K * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clusters_gpu, h_clusters_gpu, rows * cols * sizeof(uint8_t), cudaMemcpyHostToDevice);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    float time_gpu, time_cpu;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    time_t start_, end_;

    // -----------------------------------------------------------------------------------------
    // CPU KMeans

    time(&start_);

    kMeansClusteringCPU(h_image, h_clusters_cpu, h_centroids_cpu, h_clusters_sizes_cpu, rows, cols, K, MAX_ITERS);

    time(&end_);
    double dif = difftime(end_, start_);
    printf("CPU time: %.1f (s) \n", dif);

    compressImage(h_reduced_image_cpu, h_clusters_cpu, h_centroids_cpu, rows, cols);

    // -----------------------------------------------------------------------------------------
    // GPU KMeans

    cudaEventRecord(start);

    kMeansClusteringGPU(
        h_image,
        h_reduced_image_gpu,
        h_clusters_gpu,
        h_centroids_gpu,
        h_clusters_sizes_gpu,
        h_changed,
        d_image,
        d_clusters_gpu,
        d_centroids_gpu,
        d_clusters_size_gpu,
        d_changed,
        d_rand,
        &gen,
        rows,
        cols,
        K
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_gpu, start, stop);
    printf("GPU time: %.1f (s) \n", time_gpu/1000.0);

    compressImage(h_reduced_image_gpu, h_clusters_gpu, h_centroids_gpu, rows, cols);

    // -----------------------------------------------------------------------------------------
    // Show the final image
    
    Mat reduced_image_cpu = Mat(rows, cols, CV_8UC3, h_reduced_image_cpu);
    Mat reduced_image_gpu = Mat(rows, cols, CV_8UC3, h_reduced_image_gpu);
    Mat matDst(Size(image.cols * 3, image.rows), image.type(), Scalar::all(0));
    Mat matRoi = matDst(Rect(0, 0, image.cols, image.rows));
    image.copyTo(matRoi);
    matRoi = matDst(Rect(image.cols, 0, image.cols, image.rows));
    reduced_image_cpu.copyTo(matRoi);
    matRoi = matDst(Rect(image.cols * 2, 0, image.cols, image.rows));
    reduced_image_gpu.copyTo(matRoi);
    imshow("Orginal vs compressed on CPU vs compressed on GPU", matDst);
    waitKey(0);
    destroyWindow("Orginal vs compressed on CPU vs compressed on GPU");

    // -----------------------------------------------------------------------------------------
    // Free the memory

    curandDestroyGenerator(gen);

    free(h_image);
    free(h_clusters_cpu);
    free(h_clusters_gpu);
    free(h_centroids_cpu);
    free(h_centroids_gpu);
    free(h_reduced_image_cpu);
    free(h_reduced_image_gpu);
    free(h_clusters_sizes_cpu);
    free(h_clusters_sizes_gpu);
    free(h_changed);

    cudaFree(d_image);
    cudaFree(d_clusters_gpu);
    cudaFree(d_centroids_gpu);
    cudaFree(d_clusters_size_gpu);
    cudaFree(d_changed);
    cudaFree(d_rand);

    cudaDeviceReset();

    return 0;
}

