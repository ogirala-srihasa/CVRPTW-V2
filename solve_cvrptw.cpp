#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "lib/cluster/clustering.h"
#include "lib/optim/pipeline.h"
#include "lib/optim/sa_optimization.h"
#include "lib/route_utils.h"
#include "lib/vrp.h"

using namespace std;

// Above this many routes a global Clarke-Wright merge (~R^3 pair evaluations,
// each building four candidate routes and re-verifying time windows) stops
// being affordable, and the merge phase falls back to merging neighbouring
// angular clusters only.
static const int GLOBAL_MERGE_ROUTE_LIMIT = 2000;

int main(int argc, char *argv[]) {
  VRP vrp;
  if (argc < 3) {
    cout << "seqCVRPTW version 4" << '\n';
    cout << "Usage: " << argv[0]
         << " toy.vrp cluster_size [sa_rm_iterations (unused)]"
         << " [sa_only_iterations] [post_opt_iterations]" << '\n';
    exit(1);
  }

  vrp.read(argv[1]);

  int cluster_size = stoi(argv[2]);
  int sa_rm_iterations = (argc >= 4) ? stoi(argv[3]) : 10000;
  int sa_only_iterations = (argc >= 5) ? stoi(argv[4]) : 10000;
  int post_opt_iterations = (argc >= 6) ? stoi(argv[5]) : 1000;

  // The construction phase now minimizes vehicles with route_min_v2, which
  // runs until an ejection fails rather than for a fixed number of iterations.
  // argv[3] is kept in place so existing scripts and queued jobs keep their
  // argument positions, but nothing reads it.
  (void)sa_rm_iterations;

#ifdef _OPENMP
  // The construction phase nests the SA loops' own parallel regions inside a
  // parallel for over clusters. Pinning one active level keeps those inner
  // regions single-threaded instead of oversubscribing threads x threads.
  omp_set_max_active_levels(1);
#endif

  chrono::steady_clock::time_point total_start = chrono::steady_clock::now();

  auto clusters = clustering_polar_fixed_size(vrp, cluster_size);

  // Customer -> cluster index. The merge phase uses this to keep each merged
  // route associated with an angular slice.
  vector<int> cluster_of(vrp.getSize(), -1);
  for (int c = 0; c < static_cast<int>(clusters.size()); c++) {
    for (node_t cust : clusters[c]) {
      cluster_of[cust] = c;
    }
  }

  // --- Phase 1: per-cluster construction and optimization ---
  chrono::steady_clock::time_point construction_start =
      chrono::steady_clock::now();
  auto grouped_routes = construction_phase(vrp, clusters, sa_only_iterations);
  chrono::steady_clock::time_point construction_end =
      chrono::steady_clock::now();

  weight_t construction_cost = 0.0;
  int construction_vehicles = 0;
  for (const auto &group : grouped_routes) {
    construction_cost += calculate_total_cost(vrp, group);
    construction_vehicles += static_cast<int>(group.size());
  }
  cout << "Construction Cost: " << construction_cost
       << " Vehicles: " << construction_vehicles << endl;

  // --- Phase 2: Clarke-Wright merge across cluster boundaries ---
  chrono::steady_clock::time_point merge_start = chrono::steady_clock::now();
  auto routes = merge_phase(vrp, std::move(grouped_routes), cluster_of,
                            GLOBAL_MERGE_ROUTE_LIMIT);
  chrono::steady_clock::time_point merge_end = chrono::steady_clock::now();

  weight_t merge_cost = calculate_total_cost(vrp, routes);
  int merge_vehicles = static_cast<int>(routes.size());
  cout << "Merge Cost: " << merge_cost << " Vehicles: " << merge_vehicles
       << endl;

  // --- Phase 3: global eject / reinsert under SA acceptance ---
  chrono::steady_clock::time_point post_opt_start = chrono::steady_clock::now();
  int post_opt_iterations_ran = 0;
  routes = post_merge_optimization(vrp, routes, post_opt_iterations,
                                   &post_opt_iterations_ran);
  chrono::steady_clock::time_point post_opt_end = chrono::steady_clock::now();

  weight_t post_opt_cost = calculate_total_cost(vrp, routes);
  int post_opt_vehicles = static_cast<int>(routes.size());
  cout << "Post-Opt Cost: " << post_opt_cost
       << " Vehicles: " << post_opt_vehicles << endl;

  chrono::steady_clock::time_point total_end = chrono::steady_clock::now();

  weight_t final_cost = calculate_total_cost(vrp, routes);
  int final_vehicles = static_cast<int>(routes.size());
  print_routes(routes);

  auto elapsed_sec = [](chrono::steady_clock::time_point from,
                        chrono::steady_clock::time_point to) {
    return static_cast<double>(
               chrono::duration_cast<chrono::nanoseconds>(to - from).count()) *
           1.E-9;
  };

  bool valid = verify_route(vrp, routes);

  cout << "Post-Opt iterations: " << post_opt_iterations_ran
       << "  Max route length: " << max_length_of_route(routes)
       << "  Valid: " << (valid ? "yes" : "no") << endl;

  // One CSV row per run. The header is written by the test scripts:
  //   File,Construction_Time,Construction_Cost,Construction_Vehicles,
  //   Merge_Time,Merge_Cost,Merge_Vehicles,
  //   PostOpt_Time,PostOpt_Cost,PostOpt_Vehicles,
  //   Final_Time,Final_Cost,Final_Vehicles,Valid
  // Final_Time is total wall clock, so the gap between it and the three phase
  // times is preprocessing (file read + clustering). The row is emitted even
  // when verification fails, with Valid=0, so a bad run is visible instead of
  // silently missing.
  cerr << fixed << setprecision(6);
  cerr << argv[1] << ","
       << elapsed_sec(construction_start, construction_end) << ","
       << construction_cost << "," << construction_vehicles << ","
       << elapsed_sec(merge_start, merge_end) << ","
       << merge_cost << "," << merge_vehicles << ","
       << elapsed_sec(post_opt_start, post_opt_end) << ","
       << post_opt_cost << "," << post_opt_vehicles << ","
       << elapsed_sec(total_start, total_end) << ","
       << final_cost << "," << final_vehicles << ","
       << (valid ? 1 : 0) << endl;

  return 0;
}
