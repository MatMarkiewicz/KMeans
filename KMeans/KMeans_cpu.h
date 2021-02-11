#pragma once

void kMeansClusteringCPU(
	uint8_t* image,
	uint8_t* clusters,
	int* centroids,
	int* clusters_sizes,
	int rows,
	int cols,
	uint8_t K,
	int max_iters
);

void compressImage(
	uint8_t* compressed_image,
	uint8_t* clusters,
	int* centroids,
	int rows,
	int cols
);