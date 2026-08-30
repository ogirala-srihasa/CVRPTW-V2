#ifndef CLARKE_WRIGHT_H
#define CLARKE_WRIGHT_H

#include <vector>

#include "../vrp.h"

std::vector<std::vector<RouteNode>>
clarke_wright_cvrptw(const VRP &vrp,
                     const std::vector<std::vector<int>> &clusters);

std::vector<std::vector<RouteNode>>
clarke_wright_cvrptw_parallel(const VRP &vrp,
                              const std::vector<std::vector<int>> &clusters);

std::vector<std::vector<RouteNode>>
clarke_wright_cvrptw_parallel_v2(const VRP &vrp,
                                 const std::vector<std::vector<int>> &clusters);

std::vector<std::vector<RouteNode>>
clarke_wright_cvrptw_distance(const VRP &vrp,
                              const std::vector<std::vector<int>> &clusters);

std::vector<std::vector<RouteNode>>
clarke_wright_cvrptw_parallel_v3(const VRP &vrp,
                                 const std::vector<std::vector<int>> &clusters);
#endif
