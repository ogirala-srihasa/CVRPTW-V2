#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "lib/clark/clarke_wright.h"
#include "lib/cluster/clustering.h"
#include "lib/optim/inter_route_optimization.h"
#include "lib/optim/intra_route_optimization.h"
#include "lib/optim/sa_optimization.h"
#include "lib/route_utils.h"
#include "lib/vrp.h"

using namespace std;

int main(int argc, char *argv[]) {
  VRP vrp;
  if (argc < 3) {
    cout << "seqCVRPTW version 3" << '\n';
    cout << "Usage: " << argv[0]
         << " toy.vrp angle_range [sa_rm_iterations] [sa_only_iterations]" << '\n';
    exit(1);
  }

  vrp.read(argv[1]);

  chrono::steady_clock::time_point total_start = chrono::steady_clock::now();
  chrono::steady_clock::time_point pre_start = chrono::steady_clock::now();

  int n_clusters;
  int sum_demand = 0;
  for (size_t i = 1; i < vrp.getSize(); i++) {
    sum_demand += vrp.node[i].demand;
  }
  n_clusters = sum_demand / vrp.getCapacity();
  (void)n_clusters;

  double angle_range = stod(argv[2]);
  int sa_rm_iterations = (argc >= 4) ? stoi(argv[3]) : 10000;
  int sa_only_iters = (argc >= 5) ? stoi(argv[4]) : 10000;

  // vector<vector<node_t>> clusters =
  //     clustering_angle_sweep_parallel(vrp, angle_range, 1000);
  vector<vector<node_t>> clusters = clustering_angle_sweep(vrp, angle_range);
  // vector<vector<node_t>> clusters = clustering_hierarchical(vrp, n_clusters);
  // vector<vector<node_t>> clusters = clustering_kmedoid(vrp, n_clusters);
  // vector<vector<node_t>> clusters = clustering_kmeans_plus_plus(vrp,
  // n_clusters); vector<vector<node_t>> clusters = clustering_k_far(vrp,
  // n_clusters);

  for (int i = 0; i < static_cast<int>(clusters.size()); i++) {
    cout << "Cluster " << i << ": ";
    for (auto node : clusters[i]) {
      cout << node << " ";
    }
    cout << endl;
  }

  chrono::steady_clock::time_point pre_end = chrono::steady_clock::now();
  chrono::steady_clock::time_point mid_start = chrono::steady_clock::now();

#ifdef USE_PARALLEL
  auto routes = clarke_wright_cvrptw_parallel(vrp, clusters);
#else
  auto routes = clarke_wright_cvrptw(vrp, clusters);
#endif

  // Below approach is giving more average distance compared to other clark &
  // wright..... auto routes = clarke_wright_cvrptw_distance(vrp, clusters);

  chrono::steady_clock::time_point mid_end = chrono::steady_clock::now();

  for (auto &route : routes) {
    route.insert(route.begin(), RouteNode(DEPOT));
    route.push_back(RouteNode(DEPOT));
    recalculate_pred_distances(vrp, route);
  }

  weight_t construction_cost = calculate_total_cost(vrp, routes);
  int construction_vehicles = static_cast<int>(routes.size());
  // double construction_max_util, construction_avg_util;
  // compute_utilization_stats(vrp, routes, construction_max_util, construction_avg_util);
  cout << "Construction Cost: " << construction_cost
       << " Vehicles: " << construction_vehicles << endl;

  // --- Phase: Inter-route optimization ---
  chrono::steady_clock::time_point inter_start = chrono::steady_clock::now();
#ifdef USE_PARALLEL
  inter_route_relocate_parallel(vrp, routes);
  inter_route_swap_parallel(vrp, routes);
  inter_route_2opt_star_parallel(vrp, routes);
#else
  inter_route_relocate(vrp, routes);
  inter_route_swap(vrp, routes);
  inter_route_2opt_star(vrp, routes);
#endif
  chrono::steady_clock::time_point inter_end = chrono::steady_clock::now();

  weight_t inter_cost = calculate_total_cost(vrp, routes);
  int inter_vehicles = static_cast<int>(routes.size());
  // double inter_max_util, inter_avg_util;
  // compute_utilization_stats(vrp, routes, inter_max_util, inter_avg_util);
  cout << "Inter-Route Opt Cost: " << inter_cost
       << " Vehicles: " << inter_vehicles << endl;

  // --- Phase: Intra-route optimization ---
  // postProcessIt expects routes WITHOUT DEPOT bookends, so strip then re-add
  chrono::steady_clock::time_point intra_start = chrono::steady_clock::now();
  for (auto &route : routes) {
    if (!route.empty() && route.front().id == DEPOT) route.erase(route.begin());
    if (!route.empty() && route.back().id == DEPOT) route.pop_back();
  }
  weight_t intra_cost;
#ifdef USE_PARALLEL
  routes = postProcessIt_parallel(vrp, routes, intra_cost);
#else
  routes = postProcessIt(vrp, routes, intra_cost);
#endif
  for (auto &route : routes) {
    route.insert(route.begin(), RouteNode(DEPOT));
    route.push_back(RouteNode(DEPOT));
    recalculate_pred_distances(vrp, route);
  }
  chrono::steady_clock::time_point intra_end = chrono::steady_clock::now();

  intra_cost = calculate_total_cost(vrp, routes);
  int intra_vehicles = static_cast<int>(routes.size());
  // double intra_max_util, intra_avg_util;
  // compute_utilization_stats(vrp, routes, intra_max_util, intra_avg_util);
  cout << "Intra-Route Opt Cost: " << intra_cost
       << " Vehicles: " << intra_vehicles << endl;

  // --- Phase: Merged SA + Route Minimization ---
  chrono::steady_clock::time_point sa_rm_start = chrono::steady_clock::now();

  int sa_rm_iterations_ran = 0;
  auto best_routes = sa_post_optimization(vrp, routes, sa_rm_iterations, &sa_rm_iterations_ran);

  chrono::steady_clock::time_point sa_rm_end = chrono::steady_clock::now();

  weight_t sa_rm_cost = calculate_total_cost(vrp, best_routes);
  int sa_rm_vehicles = static_cast<int>(best_routes.size());
  cout << "SA+RM Cost: " << sa_rm_cost
       << " Vehicles: " << sa_rm_vehicles << endl;

  // --- Phase: SA-Only Optimization ---
  chrono::steady_clock::time_point sa_only_start = chrono::steady_clock::now();

  int sa_only_iterations_ran = 0;
  best_routes = sa_only_optimization(vrp, best_routes, sa_only_iters, &sa_only_iterations_ran);

  chrono::steady_clock::time_point sa_only_end = chrono::steady_clock::now();

  weight_t sa_only_cost = calculate_total_cost(vrp, best_routes);
  int sa_only_vehicles = static_cast<int>(best_routes.size());
  cout << "SA-Only Cost: " << sa_only_cost
       << " Vehicles: " << sa_only_vehicles << endl;
  chrono::steady_clock::time_point total_end = chrono::steady_clock::now();

  weight_t final_cost = calculate_total_cost(vrp, best_routes);
  int final_vehicles = static_cast<int>(best_routes.size());
  print_routes(best_routes);

  auto ns_to_sec = [](chrono::nanoseconds ns) {
    return static_cast<double>(ns.count()) * 1.E-9;
  };

  if (verify_route(vrp, best_routes)) {
    cerr << "File: " << argv[1] << " ";
    cerr << "Preprocessing_Time: "
         << ns_to_sec(chrono::duration_cast<chrono::nanoseconds>(pre_end - pre_start))
         << " s ";
    cerr << "Construction_Time: "
         << ns_to_sec(chrono::duration_cast<chrono::nanoseconds>(mid_end - mid_start))
         << " s ";
    cerr << "Construction_Cost: " << construction_cost << " ";
    cerr << "Construction_Vehicles: " << construction_vehicles << " ";
    // cerr << "Construction_MaxUtil: " << construction_max_util << " ";
    // cerr << "Construction_AvgUtil: " << construction_avg_util << " ";
    cerr << "InterRoute_Time: "
         << ns_to_sec(chrono::duration_cast<chrono::nanoseconds>(inter_end - inter_start))
         << " s ";
    cerr << "InterRoute_Cost: " << inter_cost << " ";
    cerr << "InterRoute_Vehicles: " << inter_vehicles << " ";
    // cerr << "InterRoute_MaxUtil: " << inter_max_util << " ";
    // cerr << "InterRoute_AvgUtil: " << inter_avg_util << " ";
    cerr << "IntraRoute_Time: "
         << ns_to_sec(chrono::duration_cast<chrono::nanoseconds>(intra_end - intra_start))
         << " s ";
    cerr << "IntraRoute_Cost: " << intra_cost << " ";
    cerr << "IntraRoute_Vehicles: " << intra_vehicles << " ";
    // cerr << "IntraRoute_MaxUtil: " << intra_max_util << " ";
    // cerr << "IntraRoute_AvgUtil: " << intra_avg_util << " ";
    cerr << "SA_RM_Time: "
         << ns_to_sec(chrono::duration_cast<chrono::nanoseconds>(sa_rm_end - sa_rm_start))
         << " s ";
    cerr << "SA_RM_Iterations: " << sa_rm_iterations_ran << " ";
    cerr << "SA_RM_Cost: " << sa_rm_cost << " ";
    cerr << "SA_RM_Vehicles: " << sa_rm_vehicles << " ";
    cerr << "SA_Only_Time: "
         << ns_to_sec(chrono::duration_cast<chrono::nanoseconds>(sa_only_end - sa_only_start))
         << " s ";
    cerr << "SA_Only_Iterations: " << sa_only_iterations_ran << " ";
    cerr << "SA_Only_Cost: " << sa_only_cost << " ";
    cerr << "SA_Only_Vehicles: " << sa_only_vehicles << " ";
    cerr << "Final_Cost: " << final_cost << " ";
    cerr << "Final_Vehicles: " << final_vehicles << " ";
    // cerr << "Final_MaxUtil: " << final_max_util << " ";
    // cerr << "Final_AvgUtil: " << final_avg_util << " ";
    cerr << "Total_Time: "
         << ns_to_sec(chrono::duration_cast<chrono::nanoseconds>(total_end - total_start))
         << " s ";
    cerr << "route_length: " << max_length_of_route(best_routes) << " ";
    cerr << "VALID" << endl;
  }

  return 0;
}
