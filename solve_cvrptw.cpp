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

  int initial_vehicles = static_cast<int>(routes.size());
  chrono::steady_clock::time_point rm_start = chrono::steady_clock::now();
  int routes_eliminated = minimize_routes(vrp, routes, 1000);
  chrono::steady_clock::time_point rm_end = chrono::steady_clock::now();

  weight_t min_cost = calculate_total_cost(vrp, routes);
  weight_t min_cost1 = min_cost;
  cout << "Total Distance: " << min_cost << endl;

  chrono::steady_clock::time_point post_start = chrono::steady_clock::now();

  int sa_iterations_ran = 0;
  auto best_routes = sa_post_optimization(vrp, routes, sa_iterations, &sa_iterations_ran);

  chrono::steady_clock::time_point post_end = chrono::steady_clock::now();
  chrono::steady_clock::time_point total_end = chrono::steady_clock::now();

  min_cost = calculate_total_cost(vrp, best_routes);
  print_routes(best_routes);

  if (verify_route(vrp, best_routes)) {
    cerr << "File: " << argv[1] << " ";
    cerr << "Preprocessing_Time: "
         << static_cast<double>(
                chrono::duration_cast<chrono::nanoseconds>(pre_end - pre_start)
                    .count() *
                1.E-9)
         << " s ";
    cerr << "Route_Construction_Time: "
         << static_cast<double>(
                chrono::duration_cast<chrono::nanoseconds>(mid_end - mid_start)
                    .count() *
                1.E-9)
         << " s ";
    cerr << "Route_Minimization_Time: "
         << static_cast<double>(
                chrono::duration_cast<chrono::nanoseconds>(rm_end - rm_start)
                    .count() *
                1.E-9)
         << " s ";
    cerr << "Routes_Eliminated: " << routes_eliminated << " ";
    cerr << "Post_Optimization_Time: "
         << static_cast<double>(chrono::duration_cast<chrono::nanoseconds>(
                                    post_end - post_start)
                                    .count() *
                                1.E-9)
         << " s ";
    cerr << "Initial_Cost: " << min_cost1 << " ";
    cerr << "Final_Cost: " << min_cost << " ";
    cerr << "Total_Time: "
         << static_cast<double>(chrono::duration_cast<chrono::nanoseconds>(
                                    total_end - total_start)
                                    .count() *
                                1.E-9)
         << " s ";
    cerr << "Vehicle_Used: " << best_routes.size() << " ";
    cerr << "SA_Iterations: " << sa_iterations_ran << " ";
    cerr << "route_length: " << max_length_of_route(best_routes) << " ";
    cerr << "VALID" << endl;
  }

  return 0;
}
