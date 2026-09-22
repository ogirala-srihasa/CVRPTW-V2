#ifndef SA_OPTIMIZATION_H
#define SA_OPTIMIZATION_H

#include <vector>

#include "../vrp.h"

std::vector<std::vector<RouteNode>> sa_post_optimization(
    const VRP &vrp,
    std::vector<std::vector<RouteNode>> routes,
    int max_iterations = 1000,
    int *iterations_ran = nullptr,
    bool enable_early_stop = true,
    bool verbose = true);

std::vector<std::vector<RouteNode>> sa_only_optimization(
    const VRP &vrp,
    std::vector<std::vector<RouteNode>> routes,
    int max_iterations = 10000,
    int *iterations_ran = nullptr,
    bool verbose = true);

std::vector<std::vector<RouteNode>> post_merge_optimization(
    const VRP &vrp,
    std::vector<std::vector<RouteNode>> routes,
    int max_iterations = 1000,
    int *iterations_ran = nullptr,
    bool verbose = true);

#endif
