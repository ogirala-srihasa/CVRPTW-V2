#ifndef INTER_ROUTE_OPTIMIZATION_H
#define INTER_ROUTE_OPTIMIZATION_H

#include <vector>

#include "../vrp.h"

void inter_route_relocate(const VRP &vrp, std::vector<std::vector<RouteNode>> &routes);
void inter_route_swap(const VRP &vrp, std::vector<std::vector<RouteNode>> &routes);
void inter_route_2opt_star(const VRP &vrp, std::vector<std::vector<RouteNode>> &routes);
void updated_relocate(const VRP &vrp, std::vector<std::vector<RouteNode>> &routes);

void inter_route_relocate_parallel(const VRP &vrp, std::vector<std::vector<RouteNode>> &routes);
void inter_route_swap_parallel(const VRP &vrp, std::vector<std::vector<RouteNode>> &routes);
void inter_route_2opt_star_parallel(const VRP &vrp, std::vector<std::vector<RouteNode>> &routes);

#endif
