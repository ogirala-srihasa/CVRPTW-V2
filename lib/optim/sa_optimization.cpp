#include "sa_optimization.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

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
// 6. Main SA post-optimization loop
//    SA acceptance is applied ONCE at the end of each iteration,
//    after all operators have been applied.
// ---------------------------------------------------------------------------
vector<vector<RouteNode>> sa_post_optimization(
    const VRP &vrp,
    vector<vector<RouteNode>> routes,
    int max_iterations) {

  cout << "\n=== Starting SA Post-Optimization (" << max_iterations
       << " iterations) ===" << endl;

  // --- SA parameters ---
  double current_cost = calculate_total_cost(vrp, routes);
  double T0 = 0.02 * current_cost;
  double alpha = 0.9995;
  double temperature = T0;

  // Best solution tracking
  auto best_routes = routes;
  double best_cost = current_cost;
  double best_cost_at_window_start = best_cost;

  // RNG for SA acceptance and ruin-and-recreate
  random_device rd;
  mt19937 rng(rd());

  const int NUM_REMOVE = 14;

  // Early stopping parameters
  const int    STAGNATION_WINDOW  = 300;
  const double STAGNATION_EPSILON = 0.001;
  const double TEMP_FLOOR_FACTOR  = 1e-5;

  cout << "  Initial cost: " << current_cost
       << "  T0: " << T0 << endl;

  bool early_stopped = false;

  for (int iter = 0; iter < max_iterations; iter++) {

    // Save state before all operators
    auto saved = routes;
    double old_cost = current_cost;

    // ---- Step 1: Ruin & Recreate ----
    ruin_and_recreate(vrp, routes, NUM_REMOVE, rng);

    // ---- Step 2: Relocate until no improving move ----
// #ifdef USE_PARALLEL
//     inter_route_relocate_parallel(vrp, routes);
// #else
//     inter_route_relocate(vrp, routes);
// #endif
    sa_best_relocate_move(vrp,routes);

    // ---- Step 3: Swap until no improving move ----
// #ifdef USE_PARALLEL
//     inter_route_swap_parallel(vrp, routes);
// #else
//     inter_route_swap(vrp, routes);
// #endif
    sa_best_swap_move(vrp,routes);

    // ---- Step 4: 2-opt* until no improving move ----
// #ifdef USE_PARALLEL
//     inter_route_2opt_star_parallel(vrp, routes);
// #else
//     inter_route_2opt_star(vrp, routes);
// #endif
    sa_best_2opt_star_move(vrp,routes);

    // ---- Step 5: Intra-route 2-opt (full multi-pass per route) ----
    sa_intra_route_2opt(vrp, routes);

    // ---- SA acceptance on the entire iteration ----
    double new_cost = calculate_total_cost(vrp, routes);
    double delta = new_cost - old_cost;

    if (sa_accept(delta, temperature, rng)) {
      current_cost = new_cost;
    } else {
      routes = std::move(saved);
      // current_cost stays as old_cost
    }

    // ---- Track global best ----
    if (current_cost < best_cost) {
      best_cost = current_cost;
      best_routes = routes;
    }

    // ---- Cool down ----
    temperature *= alpha;

    // ---- Early stopping: temperature floor ----
    if (temperature < TEMP_FLOOR_FACTOR * current_cost) {
      cout << "  Early stop (temperature floor) at iteration " << (iter + 1)
           << ": T=" << temperature << endl;
      early_stopped = true;
      break;
    }

    // ---- Early stopping: stagnation over window ----
    if ((iter + 1) % STAGNATION_WINDOW == 0) {
      double relative_improvement =
          (best_cost_at_window_start - best_cost) / best_cost_at_window_start;
      if (relative_improvement < STAGNATION_EPSILON) {
        cout << "  Early stop (stagnation) at iteration " << (iter + 1)
             << ": relative improvement " << relative_improvement
             << " over last " << STAGNATION_WINDOW << " iterations" << endl;
        early_stopped = true;
        break;
      }
      best_cost_at_window_start = best_cost;
    }

    // ---- Progress log every iteration ----
    // cout << "SA_ITER " << (iter + 1)
    //      << " current=" << current_cost
    //      << " best=" << best_cost
    //      << " T=" << temperature
    //      << " vehicles=" << routes.size() << endl;
  }

  cout << "=== SA " << (early_stopped ? "early-stopped" : "complete")
       << ". Best cost: " << best_cost
       << "  Vehicles: " << best_routes.size() << " ===" << endl;

  return best_routes;
}
