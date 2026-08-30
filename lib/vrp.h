#ifndef VRP_H
#define VRP_H

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

using point_t = double;
using weight_t = double;
using demand_t = double;
using node_t = int;
using tw_t = unsigned int;

const node_t DEPOT = 0;

class Edge {
 public:
  node_t to;
  weight_t length;

  Edge();
  Edge(node_t t, weight_t l);
  bool operator<(const Edge &e);
};

class Point {
 public:
  point_t x;
  point_t y;
  demand_t demand;
  tw_t earlyTime;
  tw_t latestTime;
  tw_t serviceTime;

  Point();
};

struct RouteNode {
  node_t id;
  weight_t dist_from_prev;

  RouteNode() : id(0), dist_from_prev(0.0) {}
  RouteNode(node_t id) : id(id), dist_from_prev(0.0) {}
  RouteNode(node_t id, weight_t d) : id(id), dist_from_prev(d) {}
  operator node_t() const { return id; }
};

class VRP {
  size_t size;
  demand_t capacity;
  std::string type;

 public:
  VRP();
  ~VRP();

  unsigned read(const std::string &filename);
  void print();

  weight_t get_dist(node_t i, node_t j) const {
    if (i == j) return 0.0;
    double dx = node[i].x - node[j].x;
    double dy = node[i].y - node[j].y;
    return std::sqrt(dx * dx + dy * dy);
  }

  size_t getSize() const;
  demand_t getCapacity() const;
  demand_t get_route_load(const std::vector<RouteNode> &route) const;

  std::vector<Point> node;
};

inline void recalculate_pred_distances(const VRP &vrp,
                                       std::vector<RouteNode> &route) {
  if (route.empty()) return;
  if (route[0].id == DEPOT) {
    route[0].dist_from_prev = 0.0;
  } else {
    route[0].dist_from_prev = vrp.get_dist(DEPOT, route[0].id);
  }
  for (size_t i = 1; i < route.size(); i++) {
    route[i].dist_from_prev = vrp.get_dist(route[i - 1].id, route[i].id);
  }
}

#endif
