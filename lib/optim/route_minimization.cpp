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
  double best_cost = calculate_total_cost(vrp, routes);
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

    // Sort candidates by route size ascending (smallest routes first)
    sort(candidate_indices.begin(), candidate_indices.end(),
         [&routes](int a, int b) {
           return routes[a].size() < routes[b].size();
         });

    // Reset eject_idx to 0 if current pair goes out of bounds
    if (eject_idx + 1 >= static_cast<int>(candidate_indices.size())) {
      eject_idx = 0;
    }

    int idx_a = candidate_indices[eject_idx];
    int idx_b = candidate_indices[eject_idx + 1];
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
    unplaced.clear();

    if (!still_unplaced.empty()) {
      vector<vector<int>> leftover_cluster = {still_unplaced};
      auto cw_routes = clarke_wright_cvrptw(vrp, leftover_cluster);
      for (auto &route : cw_routes) {
        route.insert(route.begin(), RouteNode(DEPOT));
        route.push_back(RouteNode(DEPOT));
        recalculate_pred_distances(vrp, route);
        routes.push_back(std::move(route));
      }
      still_unplaced.clear();
    }

    int current_count = static_cast<int>(routes.size());
    double current_cost = calculate_total_cost(vrp, routes);
    if (current_count < best_count ||
        (current_count == best_count && current_cost < best_cost)) {
      best_count = current_count;
      best_cost = current_cost;
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

// ---------------------------------------------------------------------------
// route_min_v2 - greedy vehicle-count minimization.
//
// Repeatedly ejects the least-loaded route and redistributes its customers
// into the remaining routes at their cheapest feasible positions. An ejection
// is committed only if EVERY customer finds a feasible slot; the first time
// one does not, the ejection is rolled back and the loop stops.
//
// Two deliberate differences from minimize_routes above:
//   - No Clarke-Wright fallback for leftovers. Building a new route to hold
//     them would put the vehicle count straight back where it started, which
//     is the opposite of the point here.
//   - No best-solution tracking. Rollback already guarantees the routes never
//     end up worse than they started, so there is nothing to track.
//
// Termination: every committed iteration removes exactly one route, so the
// loop runs at most routes.size() - 1 times before the size guard stops it.
//
// Stopping on the first failure is a heuristic, not a proof that no route can
// be dissolved. A fuller route further down the ordering might still have been
// removable - its customers could have looser time windows than the ones that
// just failed. Trying the next route instead of breaking would eliminate more
// vehicles at proportionally more search cost.
//
// This trades distance for vehicles. Absorbing another route's customers
// almost always lengthens the routes that take them, and no cost check gates
// the move - the only question asked is whether the vehicle can be removed
// while keeping every route feasible.
//
// Returns the number of routes eliminated.
// ---------------------------------------------------------------------------
int route_min_v2(const VRP &vrp,
                 vector<vector<RouteNode>> &routes,
                 bool verbose) {

  int initial_count = static_cast<int>(routes.size());

  if (verbose) {
    cout << "\n=== Starting Route Minimization v2 (" << initial_count
         << " routes) ===" << endl;
  }

  while (true) {
    // With one route left there is nothing to absorb its customers.
    if (routes.size() < 2) {
      break;
    }

    // Sort by load ascending and attack the emptiest route: fewest customers
    // to rehome, best chance of fitting them elsewhere. Re-sorted every pass
    // because a successful redistribution makes every surviving route fuller,
    // so the previous ordering no longer has the emptiest route first.
    sort(routes.begin(), routes.end(),
         [&vrp](const vector<RouteNode> &a, const vector<RouteNode> &b) {
           return vrp.get_route_load(a) < vrp.get_route_load(b);
         });

    // Snapshot for rollback - the ejection is committed only if every
    // customer is successfully rehomed.
    auto saved = routes;

    // Routes are [DEPOT, c1, ..., cn, DEPOT], so the customers are the
    // interior nodes. An already-empty route yields nothing here and is
    // simply removed, which is a legitimate elimination.
    vector<node_t> unplaced;
    for (int j = 1; j < static_cast<int>(routes[0].size()) - 1; j++) {
      unplaced.push_back(routes[0][j].id);
    }
    routes.erase(routes.begin());

    // Sort unplaced by time-window tightness: tightest first, since those
    // have the fewest feasible slots and should get first pick.
    sort(unplaced.begin(), unplaced.end(),
         [&vrp](node_t a, node_t b) {
           tw_t slack_a = vrp.node[a].latestTime - vrp.node[a].earlyTime;
           tw_t slack_b = vrp.node[b].latestTime - vrp.node[b].earlyTime;
           return slack_a < slack_b;
         });

    bool all_placed = true;

    // routes.size() does not change during reinsertion - customers are only
    // inserted into existing routes, never appended as new ones.
    int num_routes = static_cast<int>(routes.size());

    for (node_t cust : unplaced) {
      int best_route = -1;
      int best_pos = -1;
      double best_insert_cost = 1e18;

      // routes and vrp are read-only inside the region. Each thread keeps its
      // own best candidate (declared inside, so private) and they are reduced
      // through the critical section. The mutation happens after the region.
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
        // One customer with nowhere feasible to go is enough to sink this
        // ejection. No point trying the rest.
        all_placed = false;
        break;
      }
    }

    if (!all_placed) {
      routes = std::move(saved);
      break;
    }
  }

  int final_count = static_cast<int>(routes.size());
  int eliminated = initial_count - final_count;

  if (verbose) {
    cout << "=== Route Minimization v2 complete. Eliminated " << eliminated
         << " routes (" << initial_count << " -> " << final_count << ") ==="
         << endl;
  }

  return eliminated;
}
