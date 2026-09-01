#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "lib/clark/clarke_wright.h"
#include "lib/cluster/clustering.h"
#include "lib/optim/inter_route_optimization.h"
#include "lib/optim/intra_route_optimization.h"
#include "lib/optim/route_minimization.h"
#include "lib/optim/sa_optimization.h"
#include "lib/route_utils.h"
#include "lib/vrp.h"

using namespace std;

int main(int argc, char *argv[]) {
  VRP vrp;
  if (argc < 3) {
    cout << "seqCVRPTW version 3" << '\n';
    cout << "Usage: " << argv[0]
         << " toy.vrp angle_range [sa_iterations]" << '\n';
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
  int sa_iterations = (argc >= 4) ? stoi(argv[3]) : 10000;

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
  cout << "Intra-Route Opt Cost: " << intra_cost
       << " Vehicles: " << intra_vehicles << endl;

  // --- Phase: Route Minimization ---
  chrono::steady_clock::time_point rm_start = chrono::steady_clock::now();
  int routes_eliminated = minimize_routes(vrp, routes, 1000);
  chrono::steady_clock::time_point rm_end = chrono::steady_clock::now();

  weight_t rm_cost = calculate_total_cost(vrp, routes);
  int rm_vehicles = static_cast<int>(routes.size());
  cout << "Route Minimization Cost: " << rm_cost
       << " Vehicles: " << rm_vehicles << endl;

  // --- Phase: SA Post-Optimization ---
  chrono::steady_clock::time_point post_start = chrono::steady_clock::now();

  int sa_iterations_ran = 0;
  auto best_routes = sa_post_optimization(vrp, routes, sa_iterations, &sa_iterations_ran);

  chrono::steady_clock::time_point post_end = chrono::steady_clock::now();
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
    cerr << "InterRoute_Time: "
         << ns_to_sec(chrono::duration_cast<chrono::nanoseconds>(inter_end - inter_start))
         << " s ";
    cerr << "InterRoute_Cost: " << inter_cost << " ";
    cerr << "InterRoute_Vehicles: " << inter_vehicles << " ";
    cerr << "IntraRoute_Time: "
         << ns_to_sec(chrono::duration_cast<chrono::nanoseconds>(intra_end - intra_start))
         << " s ";
    cerr << "IntraRoute_Cost: " << intra_cost << " ";
    cerr << "IntraRoute_Vehicles: " << intra_vehicles << " ";
    cerr << "RouteMin_Time: "
         << ns_to_sec(chrono::duration_cast<chrono::nanoseconds>(rm_end - rm_start))
         << " s ";
    cerr << "RouteMin_Cost: " << rm_cost << " ";
    cerr << "RouteMin_Vehicles: " << rm_vehicles << " ";
    cerr << "Routes_Eliminated: " << routes_eliminated << " ";
    cerr << "SA_Time: "
         << ns_to_sec(chrono::duration_cast<chrono::nanoseconds>(post_end - post_start))
         << " s ";
    cerr << "SA_Iterations: " << sa_iterations_ran << " ";
    cerr << "Final_Cost: " << final_cost << " ";
    cerr << "Final_Vehicles: " << final_vehicles << " ";
    cerr << "Total_Time: "
         << ns_to_sec(chrono::duration_cast<chrono::nanoseconds>(total_end - total_start))
         << " s ";
    cerr << "route_length: " << max_length_of_route(best_routes) << " ";
    cerr << "VALID" << endl;
  }

  return 0;
}
