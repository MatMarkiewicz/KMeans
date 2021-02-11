#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

int dist(uint8_t* image, int* centroids, int i, int j) {
	int val;
	int dist = 0;
	for (int k = 0; k < 3; k++) {
		val = ((int)image[3 * i + k] - centroids[3 * j + k]);
		dist += val*val;
	}
	return dist;
};

void kMeansClusteringCPU(
	uint8_t* image,
	uint8_t* clusters,
	int* centroids,
	int* clusters_sizes,
	int rows,
	int cols,
	uint8_t K,
	int max_iters
){
	bool changed;
	for (int t = 0; t < max_iters; t++) {

		// clustering update
		changed = false;
		uint8_t act_best;
		int act_dist, new_dist;
		for (int i = 0; i < rows * cols; i++) {
			act_best = clusters[i];
			act_dist = dist(image, centroids, i, act_best);
			for (int j = 0; j < K; j++) {
				new_dist = dist(image, centroids, i, j);
				if (new_dist < act_dist) {
					changed = true;
					act_best = (uint8_t)j;
					act_dist = new_dist;
				}
			}
			clusters[i] = act_best;
		}

		if (t>0 && !changed) {
			break;
		}

		// centroids update
		memset(centroids, 0, K * 3 * sizeof(int));
		memset(clusters_sizes, 0, K * sizeof(int));
		int cluster, cluster_size;
		for (int i = 0; i < rows * cols; i++) {
			cluster = (int)clusters[i];
			clusters_sizes[cluster]++;
			for (int j = 0; j < 3; j++) {
				centroids[3 * cluster + j] += (int)image[3 * i + j];
			}
		}

		for (int i = 0; i < K; i++) {
			cluster_size = clusters_sizes[i];
			if (cluster_size > 0) {
				centroids[3 * i] /= cluster_size;
				centroids[3 * i + 1] /= cluster_size;
				centroids[3 * i + 2] /= cluster_size;
			}
			else {
				centroids[3 * i] = rand() % 256;
				centroids[3 * i + 1] = rand() % 256;
				centroids[3 * i + 2] = rand() % 256;
			}
		}
	}
};

void compressImage(
	uint8_t* compressed_image,
	uint8_t* clusters,
	int* centroids,
	int rows,
	int cols
){
	int cluster;
	for (int i = 0; i < rows * cols; i++) {
		for (int j = 0; j < 3; j++) {
			cluster = (int)clusters[i];
			compressed_image[3 * i + j] = centroids[3 * cluster + j];
		}
	}
};