#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <vector>

#include "../vrp.h"

std::vector<std::vector<int>> clustering_sweep(const VRP &vrp);
std::vector<std::vector<int>> clustering_angle_sweep(const VRP &vrp,
                                                     double angle_range);

std::vector<std::vector<int>> clustering_angle_sweep_parallel(
    const VRP &vrp, double angle_range, int num_trials = 100);

std::vector<std::vector<int>> clustering_hierarchical(const VRP &vrp, int k);
std::vector<std::vector<int>> clustering_kmeans_plus_plus(const VRP &vrp, int k);
std::vector<std::vector<int>> clustering_k_far(const VRP &vrp, int k);
std::vector<std::vector<int>> clustering_kmedoid(const VRP &vrp, int k);


#endif
