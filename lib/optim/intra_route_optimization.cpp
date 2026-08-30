#include "intra_route_optimization.h"

#include <cfloat>
#include <iostream>
#include <vector>

#include "../route_utils.h"

using namespace std;

void tsp_approx(const VRP &vrp,
                vector<node_t> &cities,
                vector<node_t> &tour,
                node_t ncities) {
  node_t ClosePt = 0;
  weight_t CloseDist;

  for (node_t i = 1; i < ncities; i++) {
    tour[i] = cities[i - 1];
  }
  tour[0] = cities[ncities - 1];

  double bestDistance = calculate_tour_distance_t(vrp, tour, ncities);

  for (node_t i = 1; i < ncities; i++) {
    weight_t ThisX = vrp.node[tour[i - 1]].x;
    weight_t ThisY = vrp.node[tour[i - 1]].y;
    CloseDist = DBL_MAX;

    for (node_t j = ncities - 1;; j--) {
      weight_t ThisDist =
          (vrp.node[tour[j]].x - ThisX) * (vrp.node[tour[j]].x - ThisX);
      if (ThisDist <= CloseDist) {
        ThisDist += (vrp.node[tour[j]].y - ThisY) * (vrp.node[tour[j]].y - ThisY);
        if (ThisDist <= CloseDist) {
          if (j < i) {
            break;
          }
          CloseDist = ThisDist;
          ClosePt = j;
        }
      }
    }

    unsigned temp = tour[i];
    tour[i] = tour[ClosePt];
    tour[ClosePt] = temp;

    double newDistance = calculate_tour_distance_t(vrp, tour, ncities);
    if (newDistance < bestDistance && verify_tour_t(vrp, tour, ncities)) {
      bestDistance = newDistance;
    } else {
      temp = tour[i];
      tour[i] = tour[ClosePt];
      tour[ClosePt] = temp;
    }
  }
}

vector<vector<RouteNode>> postprocess_tsp_approx(
    const VRP &vrp, vector<vector<RouteNode>> &solRoutes) {
  vector<vector<RouteNode>> modifiedRoutes;

  unsigned nroutes = solRoutes.size();
  for (unsigned i = 0; i < nroutes; ++i) {
    unsigned sz = solRoutes[i].size();
    vector<node_t> cities(sz + 1);
    vector<node_t> tour(sz + 1);

    for (unsigned j = 0; j < sz; ++j) {
      cities[j] = solRoutes[i][j];
    }
    cities[sz] = 0;

    tsp_approx(vrp, cities, tour, sz + 1);

    vector<RouteNode> curr_route;
    for (unsigned kk = 1; kk < sz + 1; ++kk) {
      curr_route.push_back(tour[kk]);
    }
    recalculate_pred_distances(vrp, curr_route);
    modifiedRoutes.push_back(curr_route);
  }
  return modifiedRoutes;
}

vector<vector<RouteNode>> postprocess_tsp_approx_parallel(
    const VRP &vrp, vector<vector<RouteNode>> &solRoutes) {

  int nroutes = solRoutes.size();

  vector<vector<RouteNode>> modifiedRoutes(nroutes);

  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < nroutes; ++i) {
    int sz = solRoutes[i].size();

    vector<node_t> cities(sz + 1);
    vector<node_t> tour(sz + 1);

    for (int j = 0; j < sz; ++j) {
      cities[j] = solRoutes[i][j];
    }
    cities[sz] = 0;

    tsp_approx(vrp, cities, tour, sz + 1);

    vector<RouteNode> curr_route;
    curr_route.reserve(sz);
    for (int kk = 1; kk < sz + 1; ++kk) {
      curr_route.push_back(tour[kk]);
    }
    recalculate_pred_distances(vrp, curr_route);

    modifiedRoutes[i] = std::move(curr_route);
  }

  return modifiedRoutes;
}



void tsp_2opt(const VRP &vrp,
              vector<node_t> &cities,
              vector<node_t> &tour,
              unsigned ncities) {
  unsigned improve = 0;

  while (improve < 2) {
    double best_distance = 0.0;
    best_distance += vrp.get_dist(DEPOT, cities[0]);
    for (unsigned jj = 1; jj < ncities; ++jj) {
      best_distance += vrp.get_dist(cities[jj - 1], cities[jj]);
    }
    best_distance += vrp.get_dist(cities[ncities - 1], DEPOT);

    for (unsigned i = 0; i < ncities - 1; ++i) {
      for (unsigned k = i + 1; k < ncities; ++k) {
        for (unsigned c = 0; c < i; ++c) {
          tour[c] = cities[c];
        }

        unsigned dec = 0;
        for (unsigned c = i; c <= k; ++c) {
          tour[c] = cities[k - dec];
          ++dec;
        }

        for (unsigned c = k + 1; c < ncities; ++c) {
          tour[c] = cities[c];
        }

        double new_distance = 0.0;
        new_distance += vrp.get_dist(DEPOT, tour[0]);
        for (unsigned jj = 1; jj < ncities; ++jj) {
          new_distance += vrp.get_dist(tour[jj - 1], tour[jj]);
        }
        new_distance += vrp.get_dist(tour[ncities - 1], DEPOT);

        vector<node_t> tmp_tour_with_depot;
        tmp_tour_with_depot.reserve(ncities + 1);
        tmp_tour_with_depot.push_back(DEPOT);
        for (unsigned t = 0; t < ncities; ++t) {
          tmp_tour_with_depot.push_back(tour[t]);
        }

        if (new_distance < best_distance &&
            verify_tour_t(vrp, tmp_tour_with_depot, ncities + 1)) {
          improve = 0;
          for (unsigned jj = 0; jj < ncities; ++jj) {
            cities[jj] = tour[jj];
          }
          best_distance = new_distance;
        }
      }
    }
    ++improve;
  }
}

vector<vector<RouteNode>> postprocess_2OPT(
    const VRP &vrp, vector<vector<RouteNode>> &final_routes) {
  vector<vector<RouteNode>> postprocessed_final_routes;

  unsigned nroutes = final_routes.size();
  for (unsigned i = 0; i < nroutes; ++i) {
    unsigned sz = final_routes[i].size();
    vector<node_t> cities(sz);
    vector<node_t> tour(sz);

    for (unsigned j = 0; j < sz; ++j) {
      cities[j] = final_routes[i][j];
    }

    if (sz > 2) {
      tsp_2opt(vrp, cities, tour, sz);
    }

    vector<RouteNode> curr_route;
    for (unsigned kk = 0; kk < sz; ++kk) {
      curr_route.push_back(cities[kk]);
    }
    recalculate_pred_distances(vrp, curr_route);
    postprocessed_final_routes.push_back(curr_route);
  }
  return postprocessed_final_routes;
}

vector<vector<RouteNode>> postprocess_2OPT_parallel(
    const VRP &vrp, vector<vector<RouteNode>> &final_routes) {

  int nroutes = final_routes.size();

  vector<vector<RouteNode>> postprocessed_final_routes(nroutes);

  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < nroutes; ++i) {
    unsigned sz = final_routes[i].size();

    vector<node_t> cities(sz);
    vector<node_t> tour(sz);

    for (unsigned j = 0; j < sz; ++j) {
      cities[j] = final_routes[i][j];
    }

    if (sz > 2) {
      tsp_2opt(vrp, cities, tour, sz);
    }

    vector<RouteNode> curr_route;
    curr_route.reserve(sz);
    for (unsigned kk = 0; kk < sz; ++kk) {
      curr_route.push_back(cities[kk]);
    }
    recalculate_pred_distances(vrp, curr_route);

    postprocessed_final_routes[i] = std::move(curr_route);
  }

  return postprocessed_final_routes;
}




vector<vector<RouteNode>> postProcessIt_parallel(
    const VRP &vrp, vector<vector<RouteNode>> &final_routes, weight_t &minCost) {
  vector<vector<RouteNode>> postprocessed_final_routes;

  auto postprocessed_final_routes1 = postprocess_tsp_approx_parallel(vrp, final_routes);
  if (verify_route_t(vrp, postprocessed_final_routes1)) {
    cout << "\nPostprocess 1 route valid" << endl;
  } else {
    cout << "\nPostprocess 1 route invalid" << endl;
  }

  auto postprocessed_final_routes2 = postprocess_2OPT_parallel(vrp, postprocessed_final_routes1);
  if (verify_route_t(vrp, postprocessed_final_routes2)) {
    cout << "Postprocess 2 route valid" << endl;
  } else {
    cout << "Postprocess 2 route invalid" << endl;
  }

  auto postprocessed_final_routes3 = postprocess_2OPT_parallel(vrp, final_routes);
  if (verify_route_t(vrp, postprocessed_final_routes3)) {
    cout << "Postprocess 3 route valid" << endl;
  } else {
    cout << "Postprocess 3 route invalid" << endl;
  }

  weight_t postprocessed_final_routes_cost = 0;
  for (unsigned zzz = 0; zzz < final_routes.size(); ++zzz) {
    vector<RouteNode> postprocessed_route2 = postprocessed_final_routes2[zzz];
    vector<RouteNode> postprocessed_route3 = postprocessed_final_routes3[zzz];

    unsigned sz2 = postprocessed_route2.size();
    unsigned sz3 = postprocessed_route3.size();

    weight_t postprocessed_route2_cost = 0.0;
    postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2[0]);
    for (unsigned jj = 1; jj < sz2; ++jj) {
      postprocessed_route2_cost +=
          vrp.get_dist(postprocessed_route2[jj - 1], postprocessed_route2[jj]);
    }
    postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2[sz2 - 1]);

    weight_t postprocessed_route3_cost = 0.0;
    postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3[0]);
    for (unsigned jj = 1; jj < sz3; ++jj) {
      postprocessed_route3_cost +=
          vrp.get_dist(postprocessed_route3[jj - 1], postprocessed_route3[jj]);
    }
    postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3[sz3 - 1]);

    if (postprocessed_route3_cost > postprocessed_route2_cost) {
      postprocessed_final_routes_cost += postprocessed_route2_cost;
      postprocessed_final_routes.push_back(postprocessed_route2);
    } else {
      postprocessed_final_routes_cost += postprocessed_route3_cost;
      postprocessed_final_routes.push_back(postprocessed_route3);
    }
  }

  minCost = postprocessed_final_routes_cost;
  return postprocessed_final_routes;
}

vector<vector<RouteNode>> postProcessIt(
    const VRP &vrp, vector<vector<RouteNode>> &final_routes, weight_t &minCost) {
  vector<vector<RouteNode>> postprocessed_final_routes;

  auto postprocessed_final_routes1 = postprocess_tsp_approx(vrp, final_routes);
  if (verify_route_t(vrp, postprocessed_final_routes1)) {
    cout << "\nPostprocess 1 route valid" << endl;
  } else {
    cout << "\nPostprocess 1 route invalid" << endl;
  }

  auto postprocessed_final_routes2 = postprocess_2OPT(vrp, postprocessed_final_routes1);
  if (verify_route_t(vrp, postprocessed_final_routes2)) {
    cout << "Postprocess 2 route valid" << endl;
  } else {
    cout << "Postprocess 2 route invalid" << endl;
  }

  auto postprocessed_final_routes3 = postprocess_2OPT(vrp, final_routes);
  if (verify_route_t(vrp, postprocessed_final_routes3)) {
    cout << "Postprocess 3 route valid" << endl;
  } else {
    cout << "Postprocess 3 route invalid" << endl;
  }

  weight_t postprocessed_final_routes_cost = 0;
  for (unsigned zzz = 0; zzz < final_routes.size(); ++zzz) {
    vector<RouteNode> postprocessed_route2 = postprocessed_final_routes2[zzz];
    vector<RouteNode> postprocessed_route3 = postprocessed_final_routes3[zzz];

    unsigned sz2 = postprocessed_route2.size();
    unsigned sz3 = postprocessed_route3.size();

    weight_t postprocessed_route2_cost = 0.0;
    postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2[0]);
    for (unsigned jj = 1; jj < sz2; ++jj) {
      postprocessed_route2_cost +=
          vrp.get_dist(postprocessed_route2[jj - 1], postprocessed_route2[jj]);
    }
    postprocessed_route2_cost += vrp.get_dist(DEPOT, postprocessed_route2[sz2 - 1]);

    weight_t postprocessed_route3_cost = 0.0;
    postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3[0]);
    for (unsigned jj = 1; jj < sz3; ++jj) {
      postprocessed_route3_cost +=
          vrp.get_dist(postprocessed_route3[jj - 1], postprocessed_route3[jj]);
    }
    postprocessed_route3_cost += vrp.get_dist(DEPOT, postprocessed_route3[sz3 - 1]);

    if (postprocessed_route3_cost > postprocessed_route2_cost) {
      postprocessed_final_routes_cost += postprocessed_route2_cost;
      postprocessed_final_routes.push_back(postprocessed_route2);
    } else {
      postprocessed_final_routes_cost += postprocessed_route3_cost;
      postprocessed_final_routes.push_back(postprocessed_route3);
    }
  }

  minCost = postprocessed_final_routes_cost;
  return postprocessed_final_routes;
}
