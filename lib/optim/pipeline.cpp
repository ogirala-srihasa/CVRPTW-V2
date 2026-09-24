#include "pipeline.h"

#include <algorithm>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../clark/clarke_wright.h"
#include "../route_utils.h"
#include "inter_route_optimization.h"
#include "intra_route_optimization.h"
#include "route_minimization.h"
#include "sa_optimization.h"

using namespace std;

// ---------------------------------------------------------------------------
// Full optimization pipeline for a single cluster.
//
// use_inner_parallel selects between the sequential and OpenMP variants of
// Clarke-Wright / inter-route / intra-route. It must be false when the caller
// is already running clusters in parallel: the SA entry points carry their own
// `omp parallel` regions, which then become nested regions and run
// single-threaded (max-active-levels is pinned to 1 in main), which is exactly
// what we want there.
// ---------------------------------------------------------------------------
static vector<vector<RouteNode>> construct_cluster_routes(
    const VRP &vrp,
    const vector<node_t> &cluster,
    int sa_only_iterations,
    bool use_inner_parallel) {

  vector<vector<node_t>> single_cluster = {cluster};

  // --- Clarke-Wright construction (returns routes without DEPOT bookends) ---
  vector<vector<RouteNode>> routes;
  if (use_inner_parallel) {
    routes = clarke_wright_cvrptw_parallel(vrp, single_cluster);
  } else {
    routes = clarke_wright_cvrptw(vrp, single_cluster);
  }

  for (auto &route : routes) {
    route.insert(route.begin(), RouteNode(DEPOT));
    route.push_back(RouteNode(DEPOT));
    recalculate_pred_distances(vrp, route);
  }

  // --- Inter-route optimization (expects DEPOT bookends) ---
  // Also erases any route that has decayed to [DEPOT, DEPOT], which is what
  // guarantees the intra-route pass below never sees an empty route.
  if (use_inner_parallel) {
    inter_route_relocate_parallel(vrp, routes);
    inter_route_swap_parallel(vrp, routes);
    inter_route_2opt_star_parallel(vrp, routes);
  } else {
    inter_route_relocate(vrp, routes);
    inter_route_swap(vrp, routes);
    inter_route_2opt_star(vrp, routes);
  }

  // --- Intra-route optimization (postProcessIt expects NO bookends) ---
  for (auto &route : routes) {
    if (!route.empty() && route.front().id == DEPOT) route.erase(route.begin());
    if (!route.empty() && route.back().id == DEPOT) route.pop_back();
  }

  weight_t intra_cost = 0.0;
  if (use_inner_parallel) {
    routes = postProcessIt_parallel(vrp, routes, intra_cost, false);
  } else {
    routes = postProcessIt(vrp, routes, intra_cost, false);
  }

  for (auto &route : routes) {
    route.insert(route.begin(), RouteNode(DEPOT));
    route.push_back(RouteNode(DEPOT));
    recalculate_pred_distances(vrp, route);
  }

  // --- Vehicle-count minimization ---
  // Greedy ejection of the emptiest route until one cannot be rehomed, in
  // place of the previous SA+RM loop. route_min_v2 self-terminates (each
  // committed pass removes exactly one route), so there is no iteration
  // budget to hand it.
  route_min_v2(vrp, routes, false);

  // --- Pure SA, early stopping enabled ---
  routes = sa_only_optimization(vrp, routes, sa_only_iterations, nullptr, false);

  return routes;
}

vector<vector<vector<RouteNode>>> construction_phase(
    const VRP &vrp,
    const vector<vector<node_t>> &clusters,
    int sa_only_iterations) {

  int num_clusters = static_cast<int>(clusters.size());
  vector<vector<vector<RouteNode>>> grouped_routes(num_clusters);

  int max_threads = 1;
#ifdef _OPENMP
  max_threads = omp_get_max_threads();
#endif

  // With enough clusters to keep every thread busy for at least two waves,
  // parallelize across clusters and run each cluster's pipeline sequentially.
  // Below that the outer loop would leave cores idle and load-imbalance badly
  // on the tail, so we walk clusters one at a time and let each phase use the
  // full thread team instead.
  bool outer_parallel = num_clusters >= 2 * max_threads;

  cout << "--- Construction phase: " << num_clusters << " clusters, "
       << (outer_parallel ? "parallel over clusters"
                          : "parallel within each cluster")
       << " ---" << endl;

  if (outer_parallel) {
    // Iteration c writes only grouped_routes[c], so there is no race and no
    // critical section is needed. vrp and clusters are read-only throughout.
    // schedule(dynamic) because per-cluster cost varies with how many routes
    // Clarke-Wright ends up building.
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < num_clusters; c++) {
      grouped_routes[c] = construct_cluster_routes(
          vrp, clusters[c], sa_only_iterations, false);
    }
  } else {
    for (int c = 0; c < num_clusters; c++) {
      grouped_routes[c] = construct_cluster_routes(
          vrp, clusters[c], sa_only_iterations, true);
    }
  }

  return grouped_routes;
}

// ---------------------------------------------------------------------------
// Merge one pair of neighbouring cluster groups, then split the result back.
//
// A merged route is handed to whichever of the two groups its first customer
// originally belonged to. That split only decides which pairs the next round
// considers -- it cannot affect feasibility, which the merge itself already
// verified.
// ---------------------------------------------------------------------------
static void merge_cluster_pair(const VRP &vrp,
                               vector<vector<vector<RouteNode>>> &grouped_routes,
                               const vector<int> &cluster_of,
                               int a,
                               int b) {
  vector<vector<RouteNode>> combined;
  combined.reserve(grouped_routes[a].size() + grouped_routes[b].size());

  for (auto &route : grouped_routes[a]) {
    combined.push_back(std::move(route));
  }
  for (auto &route : grouped_routes[b]) {
    combined.push_back(std::move(route));
  }

  auto merged = clarke_wright_merge_routes(vrp, std::move(combined));

  grouped_routes[a].clear();
  grouped_routes[b].clear();

  for (auto &route : merged) {
    // route is [DEPOT, c1, ..., cn, DEPOT], so route[1] is its first customer
    int owner = (route.size() > 2) ? cluster_of[route[1].id] : a;
    if (owner == b) {
      grouped_routes[b].push_back(std::move(route));
    } else {
      grouped_routes[a].push_back(std::move(route));
    }
  }
}

vector<vector<RouteNode>> merge_phase(
    const VRP &vrp,
    vector<vector<vector<RouteNode>>> grouped_routes,
    const vector<int> &cluster_of,
    int global_merge_route_limit) {

  int num_groups = static_cast<int>(grouped_routes.size());

  int total_routes = 0;
  for (const auto &group : grouped_routes) {
    total_routes += static_cast<int>(group.size());
  }

  // --- Global merge: every route against every other ---
  if (total_routes <= global_merge_route_limit || num_groups < 2) {
    cout << "--- Merge phase: global merge over " << total_routes
         << " routes ---" << endl;

    vector<vector<RouteNode>> all_routes;
    all_routes.reserve(total_routes);
    for (auto &group : grouped_routes) {
      for (auto &route : group) {
        all_routes.push_back(std::move(route));
      }
    }

    return clarke_wright_merge_routes_parallel(vrp, std::move(all_routes));
  }

  // --- Adjacent-boundary merge ---
  // Clusters are contiguous angular slices, and the savings term
  // d(0,i) + d(0,j) - d(i,j) is only positive when i and j are close as seen
  // from the depot. Merges between non-adjacent slices would be evaluated and
  // rejected, so restricting to neighbours costs nothing and turns an O(R^3)
  // global scan into a bounded one.
  cout << "--- Merge phase: adjacent-boundary merge over " << total_routes
       << " routes in " << num_groups << " clusters ---" << endl;

  // Round 0 takes the even boundaries (0,1) (2,3) ..., round 1 the odd ones
  // (1,2) (3,4) ... Within a round the pairs touch disjoint groups, so the
  // parallel for is race-free and needs no critical section. The inner merge
  // is the sequential variant on purpose: parallelism already lives at the
  // pair level, so this never opens a nested region.
  for (int round = 0; round < 2; round++) {
    #pragma omp parallel for schedule(dynamic)
    for (int a = round; a < num_groups - 1; a += 2) {
      merge_cluster_pair(vrp, grouped_routes, cluster_of, a, a + 1);
    }
  }

  // Angle is circular, so (num_groups - 1, 0) is a real boundary that neither
  // round above covers, for either parity of num_groups. It is a single pair,
  // so it runs on its own.
  if (num_groups > 2) {
    merge_cluster_pair(vrp, grouped_routes, cluster_of, num_groups - 1, 0);
  }

  vector<vector<RouteNode>> merged_routes;
  merged_routes.reserve(total_routes);
  for (auto &group : grouped_routes) {
    for (auto &route : group) {
      merged_routes.push_back(std::move(route));
    }
  }

  return merged_routes;
}
