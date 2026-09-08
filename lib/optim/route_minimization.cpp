#include "route_minimization.h"

#include <algorithm>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../clark/clarke_wright.h"
#include "../route_utils.h"

using namespace std;

static void build_greedy_routes(const VRP &vrp,
                                vector<vector<RouteNode>> &routes,
                                vector<node_t> &customers) {
  if (customers.empty()) return;

  sort(customers.begin(), customers.end(),
       [&vrp](node_t a, node_t b) {
         tw_t slack_a = vrp.node[a].latestTime - vrp.node[a].earlyTime;
         tw_t slack_b = vrp.node[b].latestTime - vrp.node[b].earlyTime;
         return slack_a < slack_b;
       });

  vector<bool> placed(customers.size(), false);

  for (size_t seed = 0; seed < customers.size(); seed++) {
    if (placed[seed]) continue;

    vector<RouteNode> new_route = {RouteNode(DEPOT), RouteNode(customers[seed]), RouteNode(DEPOT)};
    placed[seed] = true;
    double route_load = vrp.node[customers[seed]].demand;

    bool inserted_any = true;
    while (inserted_any) {
      inserted_any = false;
      int best_cust_idx = -1;
      int best_pos = -1;
      double best_cost = 1e18;

      for (size_t c = 0; c < customers.size(); c++) {
        if (placed[c]) continue;
        node_t cust = customers[c];

        if (route_load + vrp.node[cust].demand > vrp.getCapacity()) continue;

        for (int j = 1; j < static_cast<int>(new_route.size()); j++) {
          node_t prev = new_route[j - 1];
          node_t next = new_route[j];

          double insert_cost = vrp.get_dist(prev, cust) +
                               vrp.get_dist(cust, next) -
                               vrp.get_dist(prev, next);

          if (insert_cost < best_cost) {
            vector<RouteNode> candidate = new_route;
            candidate.insert(candidate.begin() + j, cust);

            if (verify_single_route(vrp, candidate)) {
              best_cost = insert_cost;
              best_cust_idx = static_cast<int>(c);
              best_pos = j;
            }
          }
        }
      }

      if (best_cust_idx >= 0) {
        new_route.insert(new_route.begin() + best_pos, customers[best_cust_idx]);
        route_load += vrp.node[customers[best_cust_idx]].demand;
        placed[best_cust_idx] = true;
        inserted_any = true;
      }
    }

    recalculate_pred_distances(vrp, new_route);
    routes.push_back(std::move(new_route));
  }

  customers.clear();
}

int minimize_routes(const VRP &vrp,
                    vector<vector<RouteNode>> &routes,
                    int max_attempts) {

  int initial_count = static_cast<int>(routes.size());
  vector<node_t> unplaced;

  cout << "\n=== Starting Route Minimization (max " << max_attempts
       << " attempts, " << initial_count << " routes) ===" << endl;

  int best_count = initial_count;
  auto best_routes = routes;

  int eject_idx = 0;

  for (int attempt = 0; attempt < max_attempts; attempt++) {
    // Collect indices of routes with actual customers (> 2 nodes)
    vector<int> candidate_indices;
    for (int i = 0; i < static_cast<int>(routes.size()); i++) {
      if (routes[i].size() > 2) {
        candidate_indices.push_back(i);
      }
    }

    if (candidate_indices.size() < 2) break;

    // Sort candidates by capacity utilization ascending (least utilized first)
    double capacity = vrp.getCapacity();
    sort(candidate_indices.begin(), candidate_indices.end(),
         [&vrp, &routes, capacity](int a, int b) {
           double util_a = vrp.get_route_load(routes[a]) / capacity;
           double util_b = vrp.get_route_load(routes[b]) / capacity;
           return util_a < util_b;
         });

    // Reset eject_idx to 0 if current pair goes out of bounds
    if (eject_idx + 1 >= static_cast<int>(candidate_indices.size())) {
      eject_idx = 0;
    }

    int idx_a = candidate_indices[eject_idx];
    int idx_b = candidate_indices[eject_idx + 1];

    double util_a = vrp.get_route_load(routes[idx_a]) / capacity * 100.0;
    double util_b = vrp.get_route_load(routes[idx_b]) / capacity * 100.0;
    if (util_a > 85.0 && util_b > 85.0) {
      if (eject_idx == 0) {
        cout << "  All routes above 85% utilization at attempt " << attempt
             << ", stopping." << endl;
        break;
      }
      eject_idx = 0;
      attempt--;
      continue;
    }

    eject_idx += 2;

    // Erase higher index first to avoid invalidating the lower one
    if (idx_a < idx_b) swap(idx_a, idx_b);

    for (int j = 1; j < static_cast<int>(routes[idx_a].size()) - 1; j++) {
      unplaced.push_back(routes[idx_a][j].id);
    }
    routes.erase(routes.begin() + idx_a);

    for (int j = 1; j < static_cast<int>(routes[idx_b].size()) - 1; j++) {
      unplaced.push_back(routes[idx_b][j].id);
    }
    routes.erase(routes.begin() + idx_b);

    // Sort unplaced by time-window tightness: tightest first
    sort(unplaced.begin(), unplaced.end(),
         [&vrp](node_t a, node_t b) {
           tw_t slack_a = vrp.node[a].latestTime - vrp.node[a].earlyTime;
           tw_t slack_b = vrp.node[b].latestTime - vrp.node[b].earlyTime;
           return slack_a < slack_b;
         });

    // Try to place all unplaced customers into existing routes
    vector<node_t> still_unplaced;
    int num_routes = static_cast<int>(routes.size());

    for (node_t cust : unplaced) {
      int best_route = -1;
      int best_pos = -1;
      double best_insert_cost = 1e18;

      #pragma omp parallel
      {
        int local_best_route = -1;
        int local_best_pos = -1;
        double local_best_cost = 1e18;

        #pragma omp for schedule(dynamic) nowait
        for (int r = 0; r < num_routes; r++) {
          const auto &route = routes[r];

          double load = vrp.get_route_load(route);
          if (load + vrp.node[cust].demand > vrp.getCapacity()) continue;

          for (int j = 1; j < static_cast<int>(route.size()); j++) {
            node_t prev = route[j - 1];
            node_t next = route[j];

            double insert_cost = vrp.get_dist(prev, cust) +
                                 vrp.get_dist(cust, next) -
                                 vrp.get_dist(prev, next);

            if (insert_cost < local_best_cost) {
              vector<RouteNode> candidate = route;
              candidate.insert(candidate.begin() + j, cust);

              if (verify_single_route(vrp, candidate)) {
                local_best_cost = insert_cost;
                local_best_route = r;
                local_best_pos = j;
              }
            }
          }
        }

        #pragma omp critical
        {
          if (local_best_cost < best_insert_cost) {
            best_insert_cost = local_best_cost;
            best_route = local_best_route;
            best_pos = local_best_pos;
          }
        }
      }

      if (best_route >= 0) {
        routes[best_route].insert(routes[best_route].begin() + best_pos, cust);
        recalculate_pred_distances(vrp, routes[best_route]);
        num_routes = static_cast<int>(routes.size());
      } else {
        still_unplaced.push_back(cust);
      }
    }

    if (!still_unplaced.empty()) {
      vector<vector<int>> leftover_cluster = {still_unplaced};
      auto cw_routes = clarke_wright_cvrptw_parallel(vrp, leftover_cluster);
      for (auto &route : cw_routes) {
        route.insert(route.begin(), RouteNode(DEPOT));
        route.push_back(RouteNode(DEPOT));
        recalculate_pred_distances(vrp, route);
        routes.push_back(std::move(route));
      }
      still_unplaced.clear();
    }
    unplaced.clear();

    // Track best solution seen across all attempts
    int current_count = static_cast<int>(routes.size());
    if (current_count < best_count) {
      best_count = current_count;
      best_routes = routes;
    }
  }

  routes = std::move(best_routes);

  int final_count = static_cast<int>(routes.size());
  int eliminated = initial_count - final_count;

  cout << "=== Route Minimization complete. Eliminated " << eliminated
       << " routes (" << initial_count << " -> " << final_count
       << "), " << max_attempts << " max attempts ===" << endl;

  return eliminated;
}
