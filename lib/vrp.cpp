#include "vrp.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

Edge::Edge() = default;

Edge::Edge(node_t t, weight_t l) : to(t), length(l) {}

bool Edge::operator<(const Edge &e) { return length < e.length; }

Point::Point() = default;

VRP::VRP() : size(0), capacity(0) {}

VRP::~VRP() = default;

size_t VRP::getSize() const { return size; }

demand_t VRP::getCapacity() const { return capacity; }

demand_t VRP::get_route_load(const vector<RouteNode> &route) const {
  demand_t load = 0.0;
  for (auto current_node : route) {
    load += node[current_node].demand;
  }
  return load;
}

unsigned VRP::read(const string &filename) {
  ifstream in(filename);
  if (!in.is_open()) {
    cerr << "Could not open the file \"" << filename << "\"" << endl;
    exit(1);
  }

  node.clear();

  string line;
  getline(in, line);
  string file_name = line;
  cout << "filename: " << file_name << endl;

  getline(in, line);
  getline(in, line);
  getline(in, line);

  int numVehicles;
  in >> numVehicles >> capacity;
  cout << "Vehicles: " << numVehicles << ", Capacity: " << capacity << endl;

  getline(in, line);
  getline(in, line);
  getline(in, line);
  getline(in, line);

  int id;
  double x, y, customer_demand, ready, due, service;
  while (in >> id >> x >> y >> customer_demand >> ready >> due >> service) {
    Point p;
    p.x = x;
    p.y = y;
    p.demand = customer_demand;
    p.earlyTime = ready;
    p.latestTime = due;
    p.serviceTime = service;
    node.push_back(p);
  }

  size = node.size();
  cout << "Total customers : " << size << endl;
  in.close();
  return capacity;
}

void VRP::print() {
  cout << "DIMENSION:" << size << '\n';
  cout << "CAPACITY:" << capacity << '\n';
  for (size_t i = 0; i < size; ++i) {
    cout << i << ':' << setw(6) << node[i].x << ' ' << setw(6) << node[i].y
         << ' ' << setw(6) << node[i].demand << endl;
  }
}
