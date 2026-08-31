#ifndef SA_OPTIMIZATION_H
#define SA_OPTIMIZATION_H

#include <vector>

#include "../vrp.h"

std::vector<std::vector<RouteNode>> sa_post_optimization(
    const VRP &vrp,
    std::vector<std::vector<RouteNode>> routes,
    int max_iterations = 10000,
    int *iterations_ran = nullptr);

#endif
