#include "clustering.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>
#include <cfloat>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct PolarCustomer {
  int id;
  double angle;
  double demand;
};

vector<vector<int>> clustering_sweep(const VRP &vrp) {
  int n = vrp.getSize();
  vector<vector<int>> clusters;
  if (n <= 1) {
    return clusters;
  }

  double depot_x = vrp.node[0].x;
  double depot_y = vrp.node[0].y;
  double max_capacity = vrp.getCapacity();

  vector<PolarCustomer> sweep_list;
  sweep_list.reserve(n - 1);

  for (int i = 1; i < n; i++) {
    double dx = vrp.node[i].x - depot_x;
    double dy = vrp.node[i].y - depot_y;
    double angle = atan2(dy, dx);
    if (angle < 0) {
      angle += 2.0 * M_PI;
    }
    sweep_list.push_back({i, angle, vrp.node[i].demand});
  }

  sort(sweep_list.begin(), sweep_list.end(),
       [](const PolarCustomer &a, const PolarCustomer &b) {
         return a.angle < b.angle;
       });

  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> dist(0, static_cast<int>(sweep_list.size()) - 1);
  int start_index = dist(gen);

  vector<int> current_cluster;
  double current_load = 0.0;
  int num_customers = static_cast<int>(sweep_list.size());

  for (int step = 0; step < num_customers; step++) {
    int actual_index = (start_index + step) % num_customers;
    const auto &cust = sweep_list[actual_index];

    if (current_load + cust.demand > max_capacity) {
      if (!current_cluster.empty()) {
        clusters.push_back(current_cluster);
      }
      current_cluster.clear();
      current_cluster.push_back(cust.id);
      current_load = cust.demand;
    } else {
      current_cluster.push_back(cust.id);
      current_load += cust.demand;
    }
  }

  if (!current_cluster.empty()) {
    clusters.push_back(current_cluster);
  }

  return clusters;
}

vector<vector<int>> clustering_angle_sweep(const VRP &vrp, double angle_range) {
  cout<<"--- Running Angle-Based Sweep Clustering with angle range: " << angle_range << " degrees ---" << endl;
  int n = vrp.getSize();
  vector<vector<int>> clusters;
  if (n <= 1) {
    return clusters;
  }

  double depot_x = vrp.node[0].x;
  double depot_y = vrp.node[0].y;

  vector<PolarCustomer> sweep_list;
  sweep_list.reserve(n - 1);

  for (int i = 1; i < n; i++) {
    double dx = vrp.node[i].x - depot_x;
    double dy = vrp.node[i].y - depot_y;
    double angle_rad = atan2(dy, dx);
    double angle_deg = angle_rad * (180.0 / M_PI);
    if (angle_deg < 0) {
      angle_deg += 360.0;
    }
    sweep_list.push_back({i, angle_deg, vrp.node[i].demand});
  }

  sort(sweep_list.begin(), sweep_list.end(),
       [](const PolarCustomer &a, const PolarCustomer &b) {
         return a.angle < b.angle;
       });

  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> dist(0, static_cast<int>(sweep_list.size()) - 1);
  int start_index = dist(gen);

  vector<int> current_cluster;
  double current_sector_start_angle = -1.0;
  int num_customers = static_cast<int>(sweep_list.size());

  for (int step = 0; step < num_customers; step++) {
    int actual_index = (start_index + step) % num_customers;
    const auto &cust = sweep_list[actual_index];

    if (current_cluster.empty()) {
      current_cluster.push_back(cust.id);
      current_sector_start_angle = cust.angle;
      continue;
    }

    double angular_diff = cust.angle - current_sector_start_angle;
    if (angular_diff < 0) {
      angular_diff += 360.0;
    }

    if (angular_diff > angle_range) {
      clusters.push_back(current_cluster);
      current_cluster.clear();
      current_cluster.push_back(cust.id);
      current_sector_start_angle = cust.angle;
    } else {
      current_cluster.push_back(cust.id);
    }
  }

  if (!current_cluster.empty()) {
    clusters.push_back(current_cluster);
  }

  return clusters;
}

vector<vector<int>> clustering_angle_sweep_parallel(const VRP &vrp,
                                                    double angle_range,
                                                    int num_trials) {
    int n = vrp.getSize();
    if (n <= 1) return {};
    if (num_trials <= 0) num_trials = 1;

    double depot_x = vrp.node[0].x;
    double depot_y = vrp.node[0].y;

    // 1. Pre-calculate and sort (Sequential, but very fast O(N log N))
    struct PolarCustomer { int id; double angle; double demand; };
    vector<PolarCustomer> sweep_list;
    sweep_list.reserve(n - 1);

    for (int i = 1; i < n; i++) {
        double dx = vrp.node[i].x - depot_x;
        double dy = vrp.node[i].y - depot_y;
        double angle_deg = atan2(dy, dx) * (180.0 / M_PI);
        if (angle_deg < 0) angle_deg += 360.0;
        sweep_list.push_back({i, angle_deg, vrp.node[i].demand});
    }

    sort(sweep_list.begin(), sweep_list.end(), [](const PolarCustomer &a, const PolarCustomer &b) {
        return a.angle < b.angle;
    });

    int num_customers = static_cast<int>(sweep_list.size());
    vector<vector<vector<int>>> trials(num_trials);

#pragma omp parallel
    {
        unsigned int seed = static_cast<unsigned int>(time(NULL));
#ifdef _OPENMP
        seed += static_cast<unsigned int>(omp_get_thread_num());
#endif
        mt19937 gen(seed);
        uniform_int_distribution<> dist(0, num_customers - 1);

#pragma omp for
        for (int t = 0; t < num_trials; t++) {
            int start_index = dist(gen);
            vector<vector<int>> local_clusters;
            vector<int> current_cluster;
            double current_sector_start_angle = -1.0;

            for (int step = 0; step < num_customers; step++) {
                int idx = (start_index + step) % num_customers;
                const auto &cust = sweep_list[idx];

                if (current_cluster.empty()) {
                    current_cluster.push_back(cust.id);
                    current_sector_start_angle = cust.angle;
                } else {
                    double diff = cust.angle - current_sector_start_angle;
                    if (diff < 0) diff += 360.0;

                    if (diff > angle_range) {
                        local_clusters.push_back(current_cluster);
                        current_cluster = {cust.id};
                        current_sector_start_angle = cust.angle;
                    } else {
                        current_cluster.push_back(cust.id);
                    }
                }
            }
            if (!current_cluster.empty()) local_clusters.push_back(current_cluster);
            trials[t] = std::move(local_clusters);
        }
    }

    auto get_size_variance = [](const vector<vector<int>>& clusters) {
        if (clusters.empty()) return 0.0;
        
        double mean = 0.0;
        for (const auto& c : clusters) mean += c.size();
        mean /= clusters.size();
        
        double variance = 0.0;
        for (const auto& c : clusters) {
            double diff = c.size() - mean;
            variance += (diff * diff);
        }
        return variance;
    };

    int best_idx = 0;
    for (int i = 1; i < num_trials; i++) {
        if (trials[i].size() < trials[best_idx].size()) {
            best_idx = i;
        } 
        else if (trials[i].size() == trials[best_idx].size()) {
            double current_variance = get_size_variance(trials[i]);
            double best_variance = get_size_variance(trials[best_idx]);
            
            if (current_variance > best_variance) {
                best_idx = i; 
                // cout << "Tie-breaker: Trial " << i << " has better balance (variance: " 
                //      << current_variance << ") than trial " << best_idx 
                //      << " (variance: " << best_variance << ")" << endl;
            }
        }
    }

    return trials[best_idx];
}

// Other Clustering Algorithm

vector<vector<int>> clustering_hierarchical(const VRP &vrp, int k) {
    int n = vrp.getSize();
    
    // Edge case safety
    if (k <= 0) return {};
    if (k >= n - 1) {
        vector<vector<int>> clusters(n - 1);
        for (int i = 1; i < n; i++) clusters[i-1].push_back(i);
        return clusters;
    }

    // --- PHASE 1: INITIALIZATION ---
    // Start with every customer as their own independent cluster
    vector<vector<int>> active_clusters;
    for (int i = 1; i < n; i++) {
        active_clusters.push_back({i});
    }

    int num_clusters = active_clusters.size();
    
    // Initialize the distance matrix between all clusters
    vector<vector<double>> dist_matrix(num_clusters, vector<double>(num_clusters, 0.0));

    for (int i = 0; i < num_clusters; i++) {
        for (int j = i + 1; j < num_clusters; j++) {
            // Initially, distance between single-node clusters is just their geographical distance
            // TIP: Use a spatio-temporal distance formula here for better VRPTW performance!
            dist_matrix[i][j] = vrp.get_dist(active_clusters[i][0], active_clusters[j][0]);
            dist_matrix[j][i] = dist_matrix[i][j];
        }
    }

    // --- PHASE 2: AGGLOMERATIVE MERGING ---
    // Keep merging the closest pair until we reach exactly 'k' clusters
    while (num_clusters > k) {
        double min_dist = numeric_limits<double>::max();
        int merge_a = -1, merge_b = -1;

        // 1. Find the two closest active clusters
        for (int i = 0; i < active_clusters.size(); i++) {
            if (active_clusters[i].empty()) continue; // Skip deactivated clusters

            for (int j = i + 1; j < active_clusters.size(); j++) {
                if (active_clusters[j].empty()) continue;

                if (dist_matrix[i][j] < min_dist) {
                    min_dist = dist_matrix[i][j];
                    merge_a = i;
                    merge_b = j;
                }
            }
        }

        // 2. Merge cluster B into cluster A
        active_clusters[merge_a].insert(
            active_clusters[merge_a].end(),
            active_clusters[merge_b].begin(),
            active_clusters[merge_b].end()
        );

        // Deactivate cluster B by emptying it
        active_clusters[merge_b].clear();
        num_clusters--;

        // 3. Update the Distance Matrix (COMPLETE LINKAGE)
        // The distance from the newly merged cluster A to any other cluster X 
        // is the MAXIMUM of (distance from old A to X) and (distance from B to X).
        for (int i = 0; i < active_clusters.size(); i++) {
            if (i == merge_a || active_clusters[i].empty()) continue;

            double new_dist = max(dist_matrix[merge_a][i], dist_matrix[merge_b][i]);
            
            dist_matrix[merge_a][i] = new_dist;
            dist_matrix[i][merge_a] = new_dist;
        }
    }

    // --- PHASE 3: CLEANUP ---
    // Extract only the 'k' remaining active clusters
    vector<vector<int>> final_clusters;
    for (const auto& cluster : active_clusters) {
        if (!cluster.empty()) {
            final_clusters.push_back(cluster);
        }
    }

    return final_clusters;
}

vector<vector<int>> clustering_kmeans_plus_plus(const VRP &vrp, int k) {
    int n = vrp.getSize();
    vector<vector<int>> clusters(k);
    vector<int> centers;
    
    // Edge case safety
    if (k <= 0) return clusters;
    if (k >= n - 1) {
        for (int i = 1; i < n; i++) clusters[i-1].push_back(i);
        return clusters;
    }

    vector<bool> is_center(n, false);
    
    // Track the shortest distance from each node to its nearest center
    vector<double> min_dist_to_center(n, numeric_limits<double>::max());

    // Setup Random Number Generator
    random_device rd;
    mt19937 gen(rd()); // Standard mersenne_twister_engine

    // --- PHASE 1: PICK FIRST CENTER RANDOMLY ---
    uniform_int_distribution<> uniform_dist(1, n - 1);
    int first_center = uniform_dist(gen);
    
    centers.push_back(first_center);
    is_center[first_center] = true;

    // --- PHASE 2: PICK REMAINING K-1 CENTERS (D^2 Weighting) ---
    for (int c = 1; c < k; c++) {
        int latest_center = centers.back();
        double sum_d_squared = 0.0;

        // 1. Update minimum distances for all non-center points
        // and calculate the sum of squared distances
        for (int i = 1; i < n; i++) {
            if (is_center[i]) continue;

            // Distance to the newly added center
            double d = vrp.get_dist(i, latest_center);
            
            // Update the shortest distance to ANY center
            if (d < min_dist_to_center[i]) {
                min_dist_to_center[i] = d;
            }

            // K-means++ uses D(x)^2 for its probability weight
            sum_d_squared += (min_dist_to_center[i] * min_dist_to_center[i]);
        }

        // 2. Generate a random target value between 0 and sum_d_squared
        uniform_real_distribution<double> real_dist(0.0, sum_d_squared);
        double random_target = real_dist(gen);

        // 3. Find the point that corresponds to this random target
        double cumulative_prob = 0.0;
        int next_center = -1;

        for (int i = 1; i < n; i++) {
            if (is_center[i]) continue;

            cumulative_prob += (min_dist_to_center[i] * min_dist_to_center[i]);
            
            if (cumulative_prob >= random_target) {
                next_center = i;
                break;
            }
        }

        // Safety fallback (in case of floating point precision issues)
        if (next_center == -1) {
            for (int i = 1; i < n; i++) {
                if (!is_center[i]) { next_center = i; break; }
            }
        }

        centers.push_back(next_center);
        is_center[next_center] = true;
    }

    // --- PHASE 3: ASSIGNMENT ---
    // Assign all customers to their nearest center to form the clusters
    for (int i = 1; i < n; i++) {
        int best_center_idx = -1;
        double min_dist = numeric_limits<double>::max();

        for (int j = 0; j < k; j++) {
            double d = vrp.get_dist(i, centers[j]);
            if (d < min_dist) {
                min_dist = d;
                best_center_idx = j;
            }
        }
        clusters[best_center_idx].push_back(i);
    }

    return clusters;
}

vector<vector<int>> clustering_k_far(const VRP &vrp, int k) {
    int n = vrp.getSize();
    vector<vector<int>> clusters(k);
    vector<int> centers;
    
    // Edge case safety
    if (k <= 0) return clusters;
    if (k >= n - 1) {
        // If K is greater than or equal to customers, everyone gets their own cluster
        for (int i = 1; i < n; i++) clusters[i-1].push_back(i);
        return clusters;
    }

    vector<bool> is_center(n, false);

    // --- PHASE 1: INITIALIZE CENTERS ---
    
    // 1. Pick the first center: The customer absolutely furthest from the Depot
    int first_center = -1;
    double max_dist = -1.0;
    
    for (int i = 1; i < n; i++) {
        double d = vrp.get_dist(0, i); 
        if (d > max_dist) {
            max_dist = d;
            first_center = i;
        }
    }
    centers.push_back(first_center);
    is_center[first_center] = true;

    // 2. Pick the remaining K-1 centers
    for (int c = 1; c < k; c++) {
        int next_center = -1;
        double max_min_dist = -1.0;

        for (int i = 1; i < n; i++) {
            if (is_center[i]) continue;

            // Find the minimum distance from customer 'i' to ANY existing center
            double min_dist_to_centers = numeric_limits<double>::max();
            for (int center : centers) {
                // TIP for VRPTW: Replace get_dist with a Spatio-Temporal distance 
                // function here if you want time windows factored into the clusters!
                double d = vrp.get_dist(i, center); 
                if (d < min_dist_to_centers) {
                    min_dist_to_centers = d;
                }
            }

            // We want the point where this minimum distance is the LARGEST
            // (i.e., the point sitting furthest in the middle of nowhere)
            if (min_dist_to_centers > max_min_dist) {
                max_min_dist = min_dist_to_centers;
                next_center = i;
            }
        }
        centers.push_back(next_center);
        is_center[next_center] = true;
    }

    // --- PHASE 2: ASSIGNMENT ---
    
    // 3. Assign all customers to their nearest center
    for (int i = 1; i < n; i++) {
        int best_center_idx = -1;
        double min_dist = numeric_limits<double>::max();

        for (int j = 0; j < k; j++) {
            double d = vrp.get_dist(i, centers[j]);
            if (d < min_dist) {
                min_dist = d;
                best_center_idx = j;
            }
        }
        // Add customer 'i' (even if they are a center) to the nearest cluster
        clusters[best_center_idx].push_back(i);
    }

    return clusters;
}

vector<vector<int>> clustering_kmedoid(const VRP &vrp, int k){
  int n=vrp.getSize()-1;
  vector<int> medoids_id;
  // Randomly select k medoids
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> dis(1, n); // assuming customer IDs are from 1 to n

  while(medoids_id.size()<k){
    int m_id=dis(gen);
    if(find(medoids_id.begin(),medoids_id.end(),m_id)==medoids_id.end()){
      medoids_id.push_back(m_id);
    }
  }

  cout<<"Initial Medoids: ";
  for(auto m:medoids_id){
    cout<<m<<" ";
  }
  cout<<endl;

  bool changed=true;
  vector<vector<int>> clusters(k);
  while(changed){
    changed=false;
    // Assignment Step
    clusters.clear();
    clusters.resize(k);
    for(int i=1;i<=n;i++){
      weight_t min_dist=DBL_MAX;
      int assigned_cluster=-1;
      for(int j=0;j<k;j++){
        weight_t dist=vrp.get_dist(i,medoids_id[j]);
        if(dist<min_dist){
          min_dist=dist;
          assigned_cluster=j;
        }
      }
      clusters[assigned_cluster].push_back(i);
    }

    // Update Step
    for(int j=0;j<k;j++){
      weight_t min_total_dist=DBL_MAX;
      int new_medoid=-1;
      for(auto candidate:clusters[j]){
        weight_t total_dist=0.0;
        for(auto point:clusters[j]){
          total_dist+=vrp.get_dist(candidate,point);
        }
        if(total_dist<min_total_dist){
          min_total_dist=total_dist;
          new_medoid=candidate;
        }
      }
      if(new_medoid!=medoids_id[j]){
        medoids_id[j]=new_medoid;
        changed=true;
      }
    }
  }

  // Output final clusters
  for(int j=0;j<k;j++){
    cout<<"Cluster "<<j+1<<" (Medoid: "<<medoids_id[j]<<"): ";
    for(auto customer:clusters[j]){
      cout<<customer<<" ";
    }
    cout<<endl;
  }
  return clusters;
}
