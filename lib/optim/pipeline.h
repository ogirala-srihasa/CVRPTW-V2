#ifndef PIPELINE_H
#define PIPELINE_H

#include <vector>

#include "../vrp.h"

// Runs the full per-cluster optimization pipeline (Clarke-Wright -> inter-route
// -> intra-route -> route_min_v2 -> SA-only) on every cluster. Routes are
// returned grouped by cluster, which the merge phase needs in order to pair up
// neighbouring angular slices.
std::vector<std::vector<std::vector<RouteNode>>> construction_phase(
    const VRP &vrp,
    const std::vector<std::vector<node_t>> &clusters,
    int sa_only_iterations);

// Merges routes that the clustering split apart. Runs a single global
// Clarke-Wright merge when the solution has at most global_merge_route_limit
// routes, and falls back to merging neighbouring clusters only above that.
std::vector<std::vector<RouteNode>> merge_phase(
    const VRP &vrp,
    std::vector<std::vector<std::vector<RouteNode>>> grouped_routes,
    const std::vector<int> &cluster_of,
    int global_merge_route_limit);

#endif
