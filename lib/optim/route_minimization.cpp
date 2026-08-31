#include "route_minimization.h"

#include <algorithm>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../route_utils.h"

using namespace std;

int minimize_routes(const VRP &vrp,
                    vector<vector<RouteNode>> &routes,
                    int max_attempts) {

  int initial_count = static_cast<int>(routes.size());
  int attempts = 0;
  vector<node_t> unplaced;

  cout << "\n=== Starting Route Minimization (max " << max_attempts
       << " attempts, " << initial_count << " routes) ===" << endl;

  while (attempts < max_attempts) {
    // Build candidate list: routes with actual customers (> 2 nodes = DEPOT + DEPOT)
    struct Candidate {
      int index;
      int customer_count;
      double demand;
    };
    vector<Candidate> candidates;

    for (int i = 0; i < static_cast<int>(routes.size()); i++) {
      if (routes[i].size() > 2) {
        candidates.push_back({
            i,
            static_cast<int>(routes[i].size()) - 2,
            vrp.get_route_load(routes[i])});
      }
    }

    if (candidates.empty()) break;

    // Sort: fewest customers first, then lowest demand
    sort(candidates.begin(), candidates.end(),
         [](const Candidate &a, const Candidate &b) {
           if (a.customer_count != b.customer_count)
             return a.customer_count < b.customer_count;
           return a.demand < b.demand;
         });

    bool made_progress = false;

    for (const auto &cand : candidates) {
      if (attempts >= max_attempts) break;
      attempts++;

      int route_idx = cand.index;

      // Extract customers from this route (skip DEPOT bookends)
      for (int j = 1; j < static_cast<int>(routes[route_idx].size()) - 1; j++) {
        unplaced.push_back(routes[route_idx][j].id);
      }

      // Remove the ejected route
      routes.erase(routes.begin() + route_idx);

      // Sort unplaced by time-window tightness: tightest first
      sort(unplaced.begin(), unplaced.end(),
           [&vrp](node_t a, node_t b) {
             tw_t slack_a = vrp.node[a].latestTime - vrp.node[a].earlyTime;
             tw_t slack_b = vrp.node[b].latestTime - vrp.node[b].earlyTime;
             return slack_a < slack_b;
           });

      // Try to place all unplaced customers
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
        } else {
          still_unplaced.push_back(cust);
        }
      }

      size_t prev_unplaced = unplaced.size();
      unplaced = std::move(still_unplaced);

      if (unplaced.size() < prev_unplaced) {
        made_progress = true;
      }

      // Indices invalidated by erase, restart candidate loop
      break;
    }

    if (!made_progress) break;
  }

  // Final cleanup: create single-customer routes for any remaining unplaced
  for (node_t cust : unplaced) {
    vector<RouteNode> new_route = {RouteNode(DEPOT), RouteNode(cust), RouteNode(DEPOT)};
    recalculate_pred_distances(vrp, new_route);
    routes.push_back(std::move(new_route));
  }

  int final_count = static_cast<int>(routes.size());
  int eliminated = initial_count - final_count;

  cout << "=== Route Minimization complete. Eliminated " << eliminated
       << " routes (" << initial_count << " -> " << final_count
       << "), " << attempts << " attempts ===" << endl;

  return eliminated;
}
