#ifndef ROUTE_MINIMIZATION_H
#define ROUTE_MINIMIZATION_H

#include <vector>

#include "../vrp.h"

int minimize_routes(const VRP &vrp,
                    std::vector<std::vector<RouteNode>> &routes,
                    int max_attempts = 1000);

#endif
