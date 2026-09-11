#include "sa_optimization.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../clark/clarke_wright.h"
#include "../route_utils.h"
#include "inter_route_optimization.h"
#include "intra_route_optimization.h"

using namespace std;

// ---------------------------------------------------------------------------
// Helper: Remove empty / trivially small routes (only DEPOT-DEPOT)
// ---------------------------------------------------------------------------
static void cleanup_routes(vector<vector<RouteNode>> &routes) {
  for (auto it = routes.begin(); it != routes.end();) {
    if (it->size() <= 2) {      // [DEPOT, DEPOT] or less
      it = routes.erase(it);
    } else {
      ++it;
    }
  }
}

// ---------------------------------------------------------------------------
// 1. Ruin & Recreate
//    Remove `num_remove` random customers and reinsert them one-by-one
//    at the cheapest feasible position across all routes.
// ---------------------------------------------------------------------------
static void ruin_and_recreate(const VRP &vrp,
                              vector<vector<RouteNode>> &routes,
                              int num_remove,
                              mt19937 &rng) {

  // --- Collect every removable (route_idx, position) pair ----------------
  struct CustPos {
    int route_idx;
    int pos;        // position inside route (skip 0 and last which are DEPOT)
  };
  vector<CustPos> removable;

  for (int r = 0; r < static_cast<int>(routes.size()); r++) {
    for (int p = 1; p < static_cast<int>(routes[r].size()) - 1; p++) {
      removable.push_back({r, p});
    }
  }

  // Cap num_remove to the number of available customers
  num_remove = min(num_remove, static_cast<int>(removable.size()));
  if (num_remove == 0) return;

  // Shuffle and pick the first num_remove entries
  shuffle(removable.begin(), removable.end(), rng);

  // --- Remove selected customers (highest position first per route) ------
  // Sort the chosen entries so we can remove from back to front without
  // invalidating indices.
  vector<CustPos> to_remove(removable.begin(), removable.begin() + num_remove);

  // Sort descending by (route_idx, pos) so erasing doesn't shift later indices
  sort(to_remove.begin(), to_remove.end(),
       [](const CustPos &a, const CustPos &b) {
         if (a.route_idx != b.route_idx) return a.route_idx > b.route_idx;
         return a.pos > b.pos;
       });

  vector<node_t> removed_customers;
  removed_customers.reserve(num_remove);

  for (const auto &cp : to_remove) {
    removed_customers.push_back(routes[cp.route_idx][cp.pos]);
    routes[cp.route_idx].erase(routes[cp.route_idx].begin() + cp.pos);
  }

  // Recalculate pred distances for modified routes
  for (auto &route : routes) {
    recalculate_pred_distances(vrp, route);
  }

  // Remove routes that became [DEPOT, DEPOT]
  cleanup_routes(routes);

  // --- Greedy reinsertion (one customer at a time) -----------------------
  for (node_t cust : removed_customers) {
    int best_route = -1;
    int best_pos = -1;
    double best_insert_cost = 1e18;

    int num_routes = static_cast<int>(routes.size());

    // Parallel search over all routes and positions
    #pragma omp parallel
    {
      int local_best_route = -1;
      int local_best_pos = -1;
      double local_best_cost = 1e18;

      #pragma omp for schedule(dynamic) nowait
      for (int r = 0; r < num_routes; r++) {
        const auto &route = routes[r];

        // Check capacity
        double load = vrp.get_route_load(route);
        if (load + vrp.node[cust].demand > vrp.getCapacity()) continue;

        // Try every insertion position (between index 1 and size-1)
        for (int j = 1; j < static_cast<int>(route.size()); j++) {
          node_t prev = route[j - 1];
          node_t next = route[j];

          double insert_cost = vrp.get_dist(prev, cust) +
                               vrp.get_dist(cust, next) -
                               vrp.get_dist(prev, next);

          if (insert_cost < local_best_cost) {
            // Build candidate route and verify feasibility
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

    // Insert at best position, or create a new route
    if (best_route >= 0) {
      routes[best_route].insert(routes[best_route].begin() + best_pos, cust);
      recalculate_pred_distances(vrp, routes[best_route]);
    } else {
      vector<RouteNode> new_route = {RouteNode(DEPOT), RouteNode(cust), RouteNode(DEPOT)};
      recalculate_pred_distances(vrp, new_route);
      routes.push_back(std::move(new_route));
    }
  }
}

// ---------------------------------------------------------------------------
// 2. Best single relocate move (parallel, no gain threshold)
//    Returns true if a move was found and applied.
// ---------------------------------------------------------------------------
static bool sa_best_relocate_move(const VRP &vrp,
                                  vector<vector<RouteNode>> &routes) {

  double global_best_gain = -1e18;
  int best_r1 = -1, best_r2 = -1;
  vector<RouteNode> best_routeA, best_routeB;

  int num_routes = static_cast<int>(routes.size());

  #pragma omp parallel
  {
    double local_best_gain = -1e18;
    int local_r1 = -1, local_r2 = -1;
    vector<RouteNode> local_routeA, local_routeB;

    #pragma omp for schedule(dynamic) collapse(2) nowait
    for (int r1 = 0; r1 < num_routes; r1++) {
      for (int r2 = 0; r2 < num_routes; r2++) {
        if (r1 == r2) continue;

        const auto &routeA = routes[r1];
        const auto &routeB = routes[r2];
        if (routeA.size() <= 2) continue;

        for (size_t i = 1; i < routeA.size() - 1; i++) {
          node_t u = routeA[i];
          node_t t = routeA[i - 1];
          node_t w = routeA[i + 1];

          double savings_A =
              vrp.get_dist(t, u) + vrp.get_dist(u, w) - vrp.get_dist(t, w);

          for (size_t j = 1; j < routeB.size(); j++) {
            node_t x = routeB[j - 1];
            node_t y = routeB[j];

            double cost_B =
                vrp.get_dist(x, u) + vrp.get_dist(u, y) - vrp.get_dist(x, y);
            double total_gain = savings_A - cost_B;

            if (total_gain > local_best_gain) {
              vector<RouteNode> new_routeA = routeA;
              vector<RouteNode> new_routeB = routeB;

              new_routeA.erase(new_routeA.begin() + i);
              new_routeB.insert(new_routeB.begin() + j, u);

              if (verify_single_route(vrp, new_routeA) &&
                  verify_single_route(vrp, new_routeB)) {
                local_best_gain = total_gain;
                local_r1 = r1;
                local_r2 = r2;
                local_routeA = std::move(new_routeA);
                local_routeB = std::move(new_routeB);
              }
            }
          }
        }
      }
    }

    #pragma omp critical
    {
      if (local_best_gain > global_best_gain) {
        global_best_gain = local_best_gain;
        best_r1 = local_r1;
        best_r2 = local_r2;
        best_routeA = std::move(local_routeA);
        best_routeB = std::move(local_routeB);
      }
    }
  }

  if (best_r1 < 0) return false;   // no feasible move found

  recalculate_pred_distances(vrp, best_routeA);
  recalculate_pred_distances(vrp, best_routeB);
  routes[best_r1] = std::move(best_routeA);
  routes[best_r2] = std::move(best_routeB);
  cleanup_routes(routes);
  return true;
}

// ---------------------------------------------------------------------------
// 3. Best single swap move (parallel, no gain threshold)
// ---------------------------------------------------------------------------
static bool sa_best_swap_move(const VRP &vrp,
                              vector<vector<RouteNode>> &routes) {

  double global_best_gain = -1e18;
  int best_r1 = -1, best_r2 = -1;
  vector<RouteNode> best_routeA, best_routeB;

  int num_routes = static_cast<int>(routes.size());

  #pragma omp parallel
  {
    double local_best_gain = -1e18;
    int local_r1 = -1, local_r2 = -1;
    vector<RouteNode> local_routeA, local_routeB;

    #pragma omp for schedule(dynamic) nowait
    for (int r1 = 0; r1 < num_routes; r1++) {
      for (int r2 = r1 + 1; r2 < num_routes; r2++) {

        const auto &routeA = routes[r1];
        const auto &routeB = routes[r2];

        if (routeA.size() <= 2 || routeB.size() <= 2) continue;

        double base_load_A = vrp.get_route_load(routeA);
        double base_load_B = vrp.get_route_load(routeB);

        for (size_t i = 1; i < routeA.size() - 1; i++) {
          node_t u = routeA[i];
          node_t t = routeA[i - 1];
          node_t w = routeA[i + 1];

          for (size_t j = 1; j < routeB.size() - 1; j++) {
            node_t v = routeB[j];
            node_t x = routeB[j - 1];
            node_t y = routeB[j + 1];

            double cost_before = vrp.get_dist(t, u) + vrp.get_dist(u, w) +
                                 vrp.get_dist(x, v) + vrp.get_dist(v, y);
            double cost_after = vrp.get_dist(t, v) + vrp.get_dist(v, w) +
                                vrp.get_dist(x, u) + vrp.get_dist(u, y);
            double total_gain = cost_before - cost_after;

            if (total_gain > local_best_gain) {
              double new_load_A =
                  base_load_A - vrp.node[u].demand + vrp.node[v].demand;
              double new_load_B =
                  base_load_B - vrp.node[v].demand + vrp.node[u].demand;

              if (new_load_A <= vrp.getCapacity() &&
                  new_load_B <= vrp.getCapacity()) {
                vector<RouteNode> new_routeA = routeA;
                vector<RouteNode> new_routeB = routeB;

                new_routeA[i] = v;
                new_routeB[j] = u;

                if (verify_single_route(vrp, new_routeA) &&
                    verify_single_route(vrp, new_routeB)) {
                  local_best_gain = total_gain;
                  local_r1 = r1;
                  local_r2 = r2;
                  local_routeA = std::move(new_routeA);
                  local_routeB = std::move(new_routeB);
                }
              }
            }
          }
        }
      }
    }

    #pragma omp critical
    {
      if (local_best_gain > global_best_gain) {
        global_best_gain = local_best_gain;
        best_r1 = local_r1;
        best_r2 = local_r2;
        best_routeA = std::move(local_routeA);
        best_routeB = std::move(local_routeB);
      }
    }
  }

  if (best_r1 < 0) return false;

  recalculate_pred_distances(vrp, best_routeA);
  recalculate_pred_distances(vrp, best_routeB);
  routes[best_r1] = std::move(best_routeA);
  routes[best_r2] = std::move(best_routeB);
  return true;
}

// ---------------------------------------------------------------------------
// 4. Best single 2-opt* move (parallel, no gain threshold)
// ---------------------------------------------------------------------------
static bool sa_best_2opt_star_move(const VRP &vrp,
                                   vector<vector<RouteNode>> &routes) {

  double global_best_gain = -1e18;
  int best_r1 = -1, best_r2 = -1;
  vector<RouteNode> best_routeA, best_routeB;

  int num_routes = static_cast<int>(routes.size());

  #pragma omp parallel
  {
    double local_best_gain = -1e18;
    int local_r1 = -1, local_r2 = -1;
    vector<RouteNode> local_routeA, local_routeB;

    #pragma omp for schedule(dynamic) nowait
    for (int r1 = 0; r1 < num_routes; r1++) {
      for (int r2 = r1 + 1; r2 < num_routes; r2++) {

        const auto &routeA = routes[r1];
        const auto &routeB = routes[r2];

        if (routeA.size() <= 2 || routeB.size() <= 2) continue;

        for (size_t i = 0; i < routeA.size() - 1; i++) {
          node_t t = routeA[i];
          node_t u = routeA[i + 1];

          for (size_t j = 0; j < routeB.size() - 1; j++) {
            node_t x = routeB[j];
            node_t v = routeB[j + 1];

            double cost_before = vrp.get_dist(t, u) + vrp.get_dist(x, v);
            double cost_after = vrp.get_dist(t, v) + vrp.get_dist(x, u);
            double total_gain = cost_before - cost_after;

            if (total_gain > local_best_gain) {
              vector<RouteNode> new_routeA;
              vector<RouteNode> new_routeB;

              new_routeA.reserve(routeA.size() + routeB.size());
              new_routeB.reserve(routeA.size() + routeB.size());

              new_routeA.insert(new_routeA.end(), routeA.begin(),
                                routeA.begin() + i + 1);
              new_routeA.insert(new_routeA.end(), routeB.begin() + j + 1,
                                routeB.end());

              new_routeB.insert(new_routeB.end(), routeB.begin(),
                                routeB.begin() + j + 1);
              new_routeB.insert(new_routeB.end(), routeA.begin() + i + 1,
                                routeA.end());

              double new_load_A = vrp.get_route_load(new_routeA);
              double new_load_B = vrp.get_route_load(new_routeB);

              if (new_load_A <= vrp.getCapacity() &&
                  new_load_B <= vrp.getCapacity()) {
                if (verify_single_route(vrp, new_routeA) &&
                    verify_single_route(vrp, new_routeB)) {
                  local_best_gain = total_gain;
                  local_r1 = r1;
                  local_r2 = r2;
                  local_routeA = std::move(new_routeA);
                  local_routeB = std::move(new_routeB);
                }
              }
            }
          }
        }
      }
    }

    #pragma omp critical
    {
      if (local_best_gain > global_best_gain) {
        global_best_gain = local_best_gain;
        best_r1 = local_r1;
        best_r2 = local_r2;
        best_routeA = std::move(local_routeA);
        best_routeB = std::move(local_routeB);
      }
    }
  }

  if (best_r1 < 0) return false;

  recalculate_pred_distances(vrp, best_routeA);
  recalculate_pred_distances(vrp, best_routeB);
  routes[best_r1] = std::move(best_routeA);
  routes[best_r2] = std::move(best_routeB);
  cleanup_routes(routes);
  return true;
}

// ---------------------------------------------------------------------------
// 5. Intra-route 2-opt: full multi-pass tsp_2opt on each route (parallel).
//    Extracts customer-only nodes, runs tsp_2opt to convergence, writes back.
//    Returns the total cost delta (negative = improvement).
// ---------------------------------------------------------------------------
static double sa_intra_route_2opt(const VRP &vrp,
                                  vector<vector<RouteNode>> &routes) {
  int num_routes = static_cast<int>(routes.size());
  vector<double> improvements(num_routes, 0.0);

  #pragma omp parallel for schedule(dynamic)
  for (int r = 0; r < num_routes; r++) {
    auto &route = routes[r];

    // Routes are [DEPOT, c1, c2, ..., cn, DEPOT]
    // Extract customer-only portion (skip first and last DEPOT)
    int sz = static_cast<int>(route.size()) - 2;
    if (sz <= 2) continue;

    vector<node_t> cities(sz);
    vector<node_t> tour(sz);
    for (int i = 0; i < sz; i++) {
      cities[i] = route[i + 1];    // skip DEPOT at index 0
    }

    // Compute cost before
    double cost_before = 0.0;
    cost_before += vrp.get_dist(DEPOT, cities[0]);
    for (int i = 1; i < sz; i++) {
      cost_before += vrp.get_dist(cities[i - 1], cities[i]);
    }
    cost_before += vrp.get_dist(cities[sz - 1], DEPOT);

    // Apply full multi-pass 2-opt
    tsp_2opt(vrp, cities, tour, static_cast<unsigned>(sz));

    // Compute cost after
    double cost_after = 0.0;
    cost_after += vrp.get_dist(DEPOT, cities[0]);
    for (int i = 1; i < sz; i++) {
      cost_after += vrp.get_dist(cities[i - 1], cities[i]);
    }
    cost_after += vrp.get_dist(cities[sz - 1], DEPOT);

    // Write back improved ordering into the route (keep DEPOT bookends)
    for (int i = 0; i < sz; i++) {
      route[i + 1] = cities[i];
    }

    // Recalculate pred distances after reordering
    recalculate_pred_distances(vrp, route);

    improvements[r] = cost_after - cost_before;
  }

  // Sum up total delta
  double total_delta = 0.0;
  for (int r = 0; r < num_routes; r++) {
    total_delta += improvements[r];
  }

  return total_delta;
}

// ---------------------------------------------------------------------------
// SA acceptance criterion
// ---------------------------------------------------------------------------
static bool sa_accept(double delta, double temperature, mt19937 &rng) {
  if (delta < 0.0) return true;   // improving move — always accept
  if (temperature <= 1e-12) return false;

  uniform_real_distribution<double> dist(0.0, 1.0);
  double probability = exp(-delta / temperature);
  return dist(rng) < probability;
}

// ---------------------------------------------------------------------------
// 6. Merged SA + Route Minimization loop
//    Each iteration: sort by utilization, eject 2 lowest-utilization routes
//    (if not 100% packed), cheapest reinsertion + C&W leftovers, then SA
//    operators (relocate, swap, 2-opt*, intra-2opt). SA acceptance on the
//    full iteration delta. Tracks best by lowest cost, vehicles as tiebreaker.
// ---------------------------------------------------------------------------
vector<vector<RouteNode>> sa_post_optimization(
    const VRP &vrp,
    vector<vector<RouteNode>> routes,
    int max_iterations,
    int *iterations_ran) {

  cout << "\n=== Starting Merged SA+RM (" << max_iterations
       << " iterations, " << routes.size() << " routes) ===" << endl;

  double current_cost = calculate_total_cost(vrp, routes);
  double T0 = 0.02 * current_cost;
  double alpha = 0.9995;
  double temperature = T0;

  auto best_routes = routes;
  double best_cost = current_cost;
  int best_vehicles = static_cast<int>(routes.size());

  random_device rd;
  mt19937 rng(rd());

  int eject_idx = 0;
  int actual_iterations = max_iterations;
  bool early_stopped = false;

  cout << "  Initial cost: " << current_cost
       << "  T0: " << T0
       << "  Vehicles: " << routes.size() << endl;

  for (int iter = 0; iter < max_iterations; iter++) {
    auto saved = routes;
    double old_cost = current_cost;

    int num_routes = static_cast<int>(routes.size());
    bool did_eject = false;

    if (num_routes >= 2) {
      // Sort routes by load ascending (= utilization ascending, capacity is constant)
      sort(routes.begin(), routes.end(),
           [&vrp](const vector<RouteNode> &a, const vector<RouteNode> &b) {
             return vrp.get_route_load(a) < vrp.get_route_load(b);
           });

      // Early exit: if the least-utilized route is at 100%, all are packed
      if (vrp.get_route_load(routes[0]) >= vrp.getCapacity()) {
        cout << "  All routes fully utilized, breaking at iteration "
             << (iter + 1) << endl;
        actual_iterations = iter + 1;
        early_stopped = true;
        break;
      }

      // Bounds check
      if (eject_idx + 1 >= num_routes) {
        eject_idx = 0;
      }

      double load_i = vrp.get_route_load(routes[eject_idx]);
      double load_i1 = vrp.get_route_load(routes[eject_idx + 1]);

      if (load_i < vrp.getCapacity() && load_i1 < vrp.getCapacity()) {
        did_eject = true;

        vector<node_t> unplaced;

        // Eject higher index first to preserve lower index
        int idx_hi = eject_idx + 1;
        int idx_lo = eject_idx;

        for (int j = 1; j < static_cast<int>(routes[idx_hi].size()) - 1; j++) {
          unplaced.push_back(routes[idx_hi][j].id);
        }
        routes.erase(routes.begin() + idx_hi);

        for (int j = 1; j < static_cast<int>(routes[idx_lo].size()) - 1; j++) {
          unplaced.push_back(routes[idx_lo][j].id);
        }
        routes.erase(routes.begin() + idx_lo);

        // Sort unplaced by time-window tightness (tightest first)
        sort(unplaced.begin(), unplaced.end(),
             [&vrp](node_t a, node_t b) {
               return (vrp.node[a].latestTime - vrp.node[a].earlyTime) <
                      (vrp.node[b].latestTime - vrp.node[b].earlyTime);
             });

        // Cheapest feasible reinsertion (parallel)
        vector<node_t> still_unplaced;
        int nr = static_cast<int>(routes.size());

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
            for (int r = 0; r < nr; r++) {
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
            nr = static_cast<int>(routes.size());
          } else {
            still_unplaced.push_back(cust);
          }
        }

        // Leftover consolidation via sequential C&W
        if (!still_unplaced.empty()) {
          vector<vector<int>> leftover_cluster = {still_unplaced};
          auto cw_routes = clarke_wright_cvrptw(vrp, leftover_cluster);
          for (auto &route : cw_routes) {
            route.insert(route.begin(), RouteNode(DEPOT));
            route.push_back(RouteNode(DEPOT));
            recalculate_pred_distances(vrp, route);
            routes.push_back(std::move(route));
          }
        }
      }
      // else: load_i or load_i1 >= capacity → skip ejection, SA operators only
    }

    // SA operators
    sa_best_relocate_move(vrp, routes);
    sa_best_swap_move(vrp, routes);
    sa_best_2opt_star_move(vrp, routes);
    sa_intra_route_2opt(vrp, routes);

    // SA acceptance on the entire iteration
    double new_cost = calculate_total_cost(vrp, routes);
    double delta = new_cost - old_cost;

    if (sa_accept(delta, temperature, rng)) {
      current_cost = new_cost;
    } else {
      routes = std::move(saved);
    }

    // Track global best: lowest cost first, vehicles as tiebreaker
    int current_vehicles = static_cast<int>(routes.size());
    if (current_cost < best_cost ||
        (current_cost == best_cost && current_vehicles < best_vehicles)) {
      best_cost = current_cost;
      best_vehicles = current_vehicles;
      best_routes = routes;
    }

    temperature *= alpha;

    // Advance eject index
    if (did_eject) {
      eject_idx += 2;
    } else if (num_routes >= 2) {
      // Skipped ejection (100% util hit) → reset for next iteration
      eject_idx = 0;
    }
  }

  if (!early_stopped) {
    actual_iterations = max_iterations;
  }

  cout << "=== SA+RM " << (early_stopped ? "early-stopped" : "complete")
       << ". Best cost: " << best_cost
       << "  Vehicles: " << best_vehicles << " ===" << endl;

  if (iterations_ran) *iterations_ran = actual_iterations;

  return best_routes;
}

// ---------------------------------------------------------------------------
// 7. Pure SA optimization loop (no route ejection)
//    Ruin-and-recreate + SA operators + SA acceptance.
//    Early stopping: stagnation (300-iter window, <0.1% improvement) and
//    temperature floor (T < 1e-5 * cost).
// ---------------------------------------------------------------------------
vector<vector<RouteNode>> sa_only_optimization(
    const VRP &vrp,
    vector<vector<RouteNode>> routes,
    int max_iterations,
    int *iterations_ran) {

  cout << "\n=== Starting SA-Only Optimization (" << max_iterations
       << " iterations, " << routes.size() << " routes) ===" << endl;

  double current_cost = calculate_total_cost(vrp, routes);
  double T0 = 0.02 * current_cost;
  double alpha = 0.9995;
  double temperature = T0;

  auto best_routes = routes;
  double best_cost = current_cost;
  double best_cost_at_window_start = best_cost;

  random_device rd;
  mt19937 rng(rd());

  const int NUM_REMOVE = 14;

  const int    STAGNATION_WINDOW  = 300;
  const double STAGNATION_EPSILON = 0.001;
  const double TEMP_FLOOR_FACTOR  = 1e-5;

  cout << "  Initial cost: " << current_cost
       << "  T0: " << T0 << endl;

  bool early_stopped = false;
  int actual_iterations = max_iterations;

  for (int iter = 0; iter < max_iterations; iter++) {
    auto saved = routes;
    double old_cost = current_cost;

    ruin_and_recreate(vrp, routes, NUM_REMOVE, rng);
    sa_best_relocate_move(vrp, routes);
    sa_best_swap_move(vrp, routes);
    sa_best_2opt_star_move(vrp, routes);
    sa_intra_route_2opt(vrp, routes);

    double new_cost = calculate_total_cost(vrp, routes);
    double delta = new_cost - old_cost;

    if (sa_accept(delta, temperature, rng)) {
      current_cost = new_cost;
    } else {
      routes = std::move(saved);
    }

    if (current_cost < best_cost) {
      best_cost = current_cost;
      best_routes = routes;
    }

    temperature *= alpha;

    if (temperature < TEMP_FLOOR_FACTOR * current_cost) {
      cout << "  SA-Only early stop (temperature floor) at iteration "
           << (iter + 1) << ": T=" << temperature << endl;
      actual_iterations = iter + 1;
      early_stopped = true;
      break;
    }

    if ((iter + 1) % STAGNATION_WINDOW == 0) {
      double relative_improvement =
          (best_cost_at_window_start - best_cost) / best_cost_at_window_start;
      if (relative_improvement < STAGNATION_EPSILON) {
        cout << "  SA-Only early stop (stagnation) at iteration " << (iter + 1)
             << ": relative improvement " << relative_improvement
             << " over last " << STAGNATION_WINDOW << " iterations" << endl;
        actual_iterations = iter + 1;
        early_stopped = true;
        break;
      }
      best_cost_at_window_start = best_cost;
    }
  }

  cout << "=== SA-Only " << (early_stopped ? "early-stopped" : "complete")
       << ". Best cost: " << best_cost
       << "  Vehicles: " << best_routes.size() << " ===" << endl;

  if (iterations_ran) *iterations_ran = actual_iterations;

  return best_routes;
}
