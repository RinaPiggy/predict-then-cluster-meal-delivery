# clustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import networkx as nx

# import packages
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
# import holidays
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from joblib import dump, load
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from quantile_forest import RandomForestQuantileRegressor
from scipy.cluster.hierarchy import dendrogram, linkage
import pickle
import numpy as np
from functools import lru_cache
from pathlib import Path
import h3
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse import csr_matrix

def variance_residual(y_true, y_pred):
    residuals = y_true - y_pred
    return round(np.std(residuals), 3)

def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
    # return mean_absolute_error(y_true, y_pred).round(3)

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))
    # return mean_squared_error(y_true, y_pred, squared=False).round(3)

def RMSLE(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_true+1) - np.log1p(y_pred+1))**2))

def MAPE(y_true, y_pred, c=1):
    return np.mean(np.abs((y_true - y_pred +c) / (y_true+c)) * 100)

def AE(y_true, y_pred):
    return np.abs(y_true - y_pred)

def prediction_evaluate(y_pred, y_true):
    mae = MAE(y_true, y_pred)
    rmse = RMSE(y_true, y_pred)
    rmsle = RMSLE(y_true, y_pred)
    # mape = MAPE(y_true, y_pred)
    # ae = AE(y_true, y_pred)
    resid_std = variance_residual(y_true, y_pred)
    return mae, rmse, rmsle, resid_std

def calculate_last_merge_distance(clustering, distance_matrix, num_current_cluster_):
  n_samples = distance_matrix.shape[0]
  # Retrieve the children attribute
  children = clustering.children_

  # Function to compute average linkage distance
  def compute_avg_linkage_dist(i, j, dist_matrix):
      cluster_i_points = get_cluster_points(i, children, n_samples)
      cluster_j_points = get_cluster_points(j, children, n_samples)
      return np.mean([dist_matrix[p1, p2] for p1 in cluster_i_points for p2 in cluster_j_points])

  # Helper function to get all points in a cluster
  def get_cluster_points(cluster_idx, children, n_samples):
      if cluster_idx < n_samples:
          return [cluster_idx]
      else:
          cluster = []
          for child in children[cluster_idx - n_samples]:
              cluster.extend(get_cluster_points(child, children, n_samples))
          return cluster

  # Compute the distances of each merge
  merge_distances = []
  for i, (child1, child2) in enumerate(children):
      dist = compute_avg_linkage_dist(child1, child2, distance_matrix)
      merge_distances.append(dist)

  # Get the distance of the last merge before termination
  if merge_distances:
      last_merge_distance = merge_distances[-(num_current_cluster_ - 1)]
      return last_merge_distance
  else:
      return None

# network of pickup zones
def create_zone_graph(edges, num_nodes=20):
    G = nx.Graph()
    for node in range(num_nodes):
        G.add_node(node)
    for edge in edges:
        G.add_edge(*edge)
    return G

def get_neighbors(G_general):
    return {node: list(G_general.neighbors(node)) for node in G_general.nodes}

def create_cluster_graph(G_zone, cluster_list):
    # cluster_list: [zone ji] for all zones i belong to cluster j
    G_cluster = nx.Graph()
    for j in range(len(cluster_list)):
        G_cluster.add_node(j)
    for i, cluster1 in enumerate(cluster_list):
        for j, cluster2 in enumerate(cluster_list):
            if cluster1 != cluster2 and any(G_zone.has_edge(node1, node2) for node1 in cluster1 for node2 in cluster2):
                G_cluster.add_edge(i, j)
    return G_cluster

def get_inner_neighbors(num_zones, cluster_list):
    # cluster_list: [zone ji] for all zones i belong to cluster j
    inner_neighbors = {}
    for i in range(num_zones):
        for cluster in cluster_list:
            if i in cluster:
                inner_neighbors[i] = cluster.copy()
                inner_neighbors[i].remove(i)
    return inner_neighbors

def get_outer_neighbors(num_zones, max_cluster_size, cluster_list, G_cluster):
    cluster_outer_neighbor_dict = get_neighbors(G_cluster)
    zone_outer_neighbor_dict = {}

    # Create a new list that only includes clusters with size less than or equal to max_cluster_size
    valid_clusters = [cluster for cluster in cluster_list if len(cluster) < max_cluster_size]

    # Update cluster_outer_neighbor_dict to only include valid clusters
    cluster_outer_neighbor_dict = {i: cluster_outer_neighbor_dict[i] for i in range(len(valid_clusters))}

    for i in range(num_zones):
        neighbors_ = []
        for c_id, cluster in enumerate(valid_clusters):
            if i in cluster:
                cluster_neighbors = cluster_outer_neighbor_dict[c_id]
                for nb_cluster_id in cluster_neighbors:
                    if nb_cluster_id in cluster_outer_neighbor_dict.keys():
                        if len(valid_clusters[nb_cluster_id]) + len(cluster) <= max_cluster_size:
                            neighbors_ += valid_clusters[nb_cluster_id]
        zone_outer_neighbor_dict[i] = list(set(neighbors_))

    return zone_outer_neighbor_dict

def create_linkage_matrix(num_zones, zone_outer_neighbor_dict, zone_inner_neighbor_dict):
    # create the potential linkage matrix
    potential_linkages = np.zeros((num_zones, num_zones))
    for zone, outer_neighbors in zone_outer_neighbor_dict.items():
        for outer_neighbor in outer_neighbors:
            potential_linkages[zone][outer_neighbor] = 1
    assert np.allclose(potential_linkages, potential_linkages.T, rtol=1e-05, atol=1e-08)
    assert np.allclose(np.diag(potential_linkages), np.zeros(num_zones), rtol=1e-05, atol=1e-08)

    # create the must linkage matrix
    must_linkages = np.zeros((num_zones, num_zones))
    for zone, inner_neighbors in zone_inner_neighbor_dict.items():
        for inner_neighbor in inner_neighbors:
            must_linkages[zone][inner_neighbor] = 1
    # the diagonal of the must linkage matrix should be zero, and the matrix should be symmetric
    assert np.allclose(must_linkages, must_linkages.T, rtol=1e-05, atol=1e-08)
    assert np.allclose(np.diag(must_linkages), np.zeros(num_zones), rtol=1e-05, atol=1e-08)

    return potential_linkages, must_linkages

# implement the weighted Euclidean distance in main code
def weighted_euclidean_distance(v1, v2, w):
    # v1, v2 and w are arrays of shape (3,0)
    # Calculate the weighted Euclidean distance
    diff = v1 - v2
    weighted_diff = w * (diff ** 2)
    weighted_distance = np.sqrt(np.sum(weighted_diff))
    return weighted_distance

def get_distance_metrix(num_zones, 
                        attributes_, 
                        potential_linkages, 
                        must_linkages, 
                        attribute_type = 'single',
                        large_value_mask = 99):
    # attributes_ = df[feature_list], a 2D array of the features of all zones, for specific time step t
    # attributes_[i]:  the feature vector of zone i
    
    # DEBUGGER
    # print(f'Attribute type (inter-function): {attribute_type}')
    # print(attributes_.shape)

    distance_matrix = np.zeros((num_zones, num_zones))
    for i in range(num_zones):
        for j in range(num_zones):
            if i != j:
                if isinstance(attribute_type, list):
                    # check whether attribute_type is a list of 3 elements:
                    if len(attribute_type) == len(attributes_[i]):
                        distance_matrix[i, j] = weighted_euclidean_distance(attributes_[i], attributes_[j], w=np.array(attribute_type))
                    else:
                        print(len(attribute_type))
                        print(len(attributes_[i]))
                        assert(len(attribute_type) == len(attributes_[i]))
                else:
                    if attribute_type == 'single':
                        distance_matrix[i, j] = np.linalg.norm(attributes_[i] - attributes_[j])
                    elif attribute_type == 'multiple':
                        distance_matrix[i, j] = weighted_euclidean_distance(attributes_[i], attributes_[j], w=np.array([1,2,1]))
    
                if potential_linkages[i, j] == 0:
                    # if two zones are impossible to link, set their distance to a large number
                    distance_matrix[i, j] = large_value_mask
                if must_linkages[i, j] == 1:
                    # if two zones must be linked, set their distance to zero
                    distance_matrix[i, j] = 0
    
    # Ensure the distance matrix is symmetric
    distance_matrix = np.minimum(distance_matrix, distance_matrix.T)

    return distance_matrix

# def similarity_constraint (distance_matrix, max_distance, large_value_mask=99):
#     '''
#     return filtered distance_matrix (numpy matrix of shape N x N, N: number of zones). 
#     If zone-wise similarity exceeds max_distance (in similarity), 
#     mask with arbitrarily high values.
#     '''
#     result = distance_matrix.copy()
#     result[result>max_distance] = large_value_mask
#     return result

def contingency_constrained_hierarchical_clustering(attributes,
                                                    zone_edges,
                                                    zone_actual_demand, 
                                                    zone_pred_demand,
                                                    num_zones = 20,
                                                    ultimate_num_clusters = 5, 
                                                    max_cluster_size = 6, 
                                                    distance_threshold_=9,
                                                    distance_measure = 'ward',
                                                    print_ = False,
                                                    attribute_type = 'single',
                                                    large_value_mask = 99):
    '''
    This function generates the clustering of the zones based on their predicted demand values,
    by the contingency constrained hierarchical clustering process

    Inputs: 
    df: the dataframe containing the predicted and previous demand values of all pick-up zones, for each time step t
    feature_list: the list of features that will be used as criterion of clustering

    Constraints:
    max_num_clusters: the maximum number of clusters that can be generated, default = 5
    min_num_clusters: the minimum number of clusters that can be generated, default = 2
    max_cluster_size: the maximum number of zones that can be in a cluster, default = 9
    min_cluster_size: the minimum number of zones that can be in a cluster, default = 1
    (contingency constraints are updated in the linkage matrix, during the clustering process)
    distance_threshold_: the maximum distance / minimal similarity between mergable zones
    distance_measure: the distance measure used in the clustering process, default = 'ward', we should perform sensitivity analysis over this parameter

    Outputs:
    clustering.labels_: the cluster label of each zone
    cluster_demand_actual: the average demand value of each cluster, calculated using actual demand values, based on the clustering result over predicted attributes
    cluster_demand_pred: the average predicted demand value of each cluster, calculated using predicted demand values, based on the clustering result over predicted attributes
    '''
    # initialize helper variables
    cluster_list= [[i] for i in range(num_zones)] # [zone ji] for all zones i belong to cluster j
    # print(cluster_list)
    cluster_sizes = np.ones(num_zones)
    # temp_max_cluster_size = np.max(cluster_sizes)
    last_merge_distance_ = 99

    # intialize the network of zones & clusters using networkx
    G_zone = create_zone_graph(zone_edges)
    # zone_geoconnected_dict = get_neighbors(G_zone)

    # Perform hierarchical clustering: 
    num_current_cluster_ = num_zones
    violations_ = False
    
    while num_current_cluster_ > ultimate_num_clusters and violations_ == False:
        num_current_cluster_ -= 1
        # print(f'Current number of clusters: {num_current_cluster_}')
        # create the cluster graph
        G_cluster = create_cluster_graph(G_zone, cluster_list)
        zone_outer_neighbor_dict = get_outer_neighbors(num_zones, max_cluster_size, cluster_list, G_cluster) # this is used to update the potential linkage matrix
        zone_inner_neighbor_dict = get_inner_neighbors(num_zones, cluster_list) # this is used to update the must linkage matrix

        # how to create the linkage matrix from zone_outer_neighbor_dict and zone_inner_neighbor_dict
        potential_linkages, must_linkages = create_linkage_matrix(num_zones, zone_outer_neighbor_dict, zone_inner_neighbor_dict)

        # Modify the distance matrix to enforce can-link constraints
        distance_matrix = get_distance_metrix(num_zones, attributes, potential_linkages, must_linkages, attribute_type, large_value_mask)
        # print('--The next smallest distance in line', linkage(distance_matrix, method='average'))
        
        clustering = AgglomerativeClustering(
            n_clusters=num_current_cluster_,
            affinity='precomputed', # we use the precomputed distance matrix
            linkage=distance_measure,
            compute_full_tree=False
        )

        # Fit the clustering
        clustering.fit(distance_matrix)

        # size of each cluster after the clustering
        cluster_sizes = np.zeros(num_zones)
        for i in range(num_zones):
            cluster_sizes[clustering.labels_[i]] += 1

        # cluster similarity threshold via last merge distance
        last_merge_distance = calculate_last_merge_distance(clustering, distance_matrix, num_current_cluster_)
        # if last_merge_distance > 10:
        #     print('last merge distance', last_merge_distance)
        #     print(cluster_sizes)
        if last_merge_distance > distance_threshold_: 
            violations_ = True

        # create cluster list, which is a 2D list of zones in each cluster
        cluster_list = [[] for _ in range(num_current_cluster_)]
        for i in range(num_zones):
            cluster_list[clustering.labels_[i]].append(i)

        # check whether zones that do not included by zone_inner_neighbor_dict and zone_outer_neighbor_dict are in the same cluster
        for i in range(num_zones):
            if violations_ == False:
                # get the list of zones in the same cluster with zone i
                cluster_i = cluster_list[clustering.labels_[i]]
                # check whether any zone j in cluster_i is neither in zone_inner_neighbor_dict nor in zone_outer_neighbor_dict
                for j in cluster_i:
                    if j != i: 
                        if j not in zone_inner_neighbor_dict[i] and j not in zone_outer_neighbor_dict[i]:
                            violations_ = True
                            break

        # revert to the previous clustering
        if violations_:
            num_current_cluster_ += 1
            if print_:
                print('Contingency violation detected, reverting to the previous clustering')
            clustering = AgglomerativeClustering(n_clusters=num_current_cluster_,
                                                affinity='precomputed', # we use the precomputed distance matrix
                                                linkage=distance_measure,
                                                compute_full_tree=False)

            # Fit the clustering
            clustering.fit(distance_matrix)

    # size of each cluster after the clustering
    cluster_sizes = np.zeros(num_current_cluster_)
    for i in range(num_zones):
        cluster_sizes[clustering.labels_[i]] += 1

    cluster_list = [[] for _ in range(num_current_cluster_)]
    for i in range(num_zones):
        cluster_list[clustering.labels_[i]].append(i)

    # for each cluster, calculate the average demand value using zone_actual_demand
    cluster_medi_demand_actual = []
    cluster_medi_demand_pred = []
    # cluster_avg_demand_actual = []
    # cluster_avg_demand_pred = []
    for cluster in cluster_list:
        # demand = 0
        # demand_pred = 0
        demand = []
        demand_pred = []
        for zone in cluster:
            # demand += zone_actual_demand[zone]
            # demand_pred += zone_pred_demand[zone]
            demand.append(zone_actual_demand[zone])
            demand_pred.append(zone_pred_demand[zone])
        # cluster_avg_demand_actual.append(demand / len(cluster))
        # cluster_avg_demand_pred.append(demand_pred / len(cluster))
        cluster_medi_demand_actual.append(np.median(demand))
        cluster_medi_demand_pred.append(np.median(demand_pred))
    
    if print_:
        print('number of clusters', num_current_cluster_)
        print('final labels', clustering.labels_)
        print('final cluster sizes', cluster_sizes)
        # print('cluster average demand (actual)', cluster_avg_demand_actual)
        # print('cluster average demand (predict)', cluster_avg_demand_pred)
        print('cluster median demand (actual)', cluster_medi_demand_actual)
        print('cluster median demand (predict)', cluster_medi_demand_pred)

    # create an array that contains the cluster average demand for each zone
    cluster_demand_actual = np.zeros(num_zones)
    cluster_demand_pred = np.zeros(num_zones)
    for i in range(num_zones):
        # print('zone', i, 'cluster', clustering.labels_[i])
        # cluster_demand_actual[i] = cluster_avg_demand_actual[clustering.labels_[i]]
        # cluster_demand_pred[i] = cluster_avg_demand_pred[clustering.labels_[i]]
        cluster_demand_actual[i] = cluster_medi_demand_actual[clustering.labels_[i]]
        cluster_demand_pred[i] = cluster_medi_demand_pred[clustering.labels_[i]]
    
    return clustering.labels_, cluster_demand_actual, cluster_demand_pred

        # # Optional: boundary smoothing: if at least 5 out of 6 neighbors of a zone belongs to the same cluster, then the zone should also belong to that cluster
        # for i in range(num_zones):
        #     if cluster_sizes[clustering.labels_[i]] < temp_max_cluster_size:
        #         neighbors = zone_geoconnected_dict[i]
        #         neighbor_clusters = [clustering.labels_[neighbor] for neighbor in neighbors]
        #         neighbor_clusters = np.array(neighbor_clusters)
        #         cluster_id = stats.mode(neighbor_clusters).mode[0]
        #         if np.sum(neighbor_clusters == cluster_id) >= 5 and cluster_id != clustering.labels_[i]:
        #             print('Boundary Smoothing!!')
        #             cluster_sizes[clustering.labels_[i]] -= 1
        #             clustering.labels_[i] = cluster_id
        #             cluster_sizes[cluster_id] += 1
         
        # print('cluster sizes',cluster_sizes)
        # print('labels', clustering.labels_)
        # print('\n')

import time

def hier_clustering_per_timestep(attributes_biglist,
                                 zone_wise_demand_actual,
                                 zone_wise_demand_pred, 
                                 zone_edges_,
                                 distance_threshold_=9,
                                 min_num_cluster = 2,
                                 max_size_per_cluster = 7,
                                 proximacy_measure = 'average',
                                 attribute_type = 'single'):
    
    # DEBUGGER check attribute_type
    print(f'Attribute type: {attribute_type}')
    print('check input attribute shape', attributes_biglist.shape)
    print('check input zone_wise_demand_pred shape', zone_wise_demand_pred.shape)

    num_timestep = len(attributes_biglist)
    list_of_labels = []
    list_of_cluster_demand_actual = []
    list_of_cluster_demand_pred = []
    # new for computational time recording
    start_time = time.time()
    for i in range(num_timestep):
        clustering_labels_, cluster_demand_actual, cluster_demand_pred =contingency_constrained_hierarchical_clustering(
                                                attributes = attributes_biglist[i],
                                                zone_edges = zone_edges_,
                                                zone_actual_demand = zone_wise_demand_actual[i], 
                                                zone_pred_demand = zone_wise_demand_pred[i],
                                                num_zones = 20,
                                                distance_threshold_ = distance_threshold_,
                                                ultimate_num_clusters = min_num_cluster, 
                                                max_cluster_size = max_size_per_cluster, 
                                                distance_measure = proximacy_measure,
                                                attribute_type = attribute_type)
        # COLLECT all the labels, cluster_actual and cluster_pred into a full length array
        list_of_labels.append(clustering_labels_) # at timestep t and for all 20 grids
        list_of_cluster_demand_actual.append(cluster_demand_actual)
        list_of_cluster_demand_pred.append(cluster_demand_pred)
    # print the time elapsed for each clustering
    print(f'Average Time elapsed for clustering at time step {i}: {(time.time() - start_time)/num_timestep:.3f} seconds')

    list_of_labels = np.array(list_of_labels).flatten()
    list_of_cluster_demand_actual = np.array(list_of_cluster_demand_actual).flatten()
    list_of_cluster_demand_pred = np.array(list_of_cluster_demand_pred).flatten()
    return list_of_labels, list_of_cluster_demand_actual, list_of_cluster_demand_pred

def clustering_evaluation(df_full, 
                          clustering_labels, 
                          cluster_demand_actual, 
                          cluster_demand_pred,
                          print_out = False):
    '''
    This universal evaluation func compares the estimated courier resource needed post clustering
    
    Measure: Error/Difference from {estimated per cluster resource (using actual demand values attributes) - estimated per cluster resource (using pred demand attributes)}

    Note that, the measure accounts for influence from
    - (1) forecasting error from the point predicted demand for each zone per 15min time window
    - (2) the clustering deviation caused by using (potentially flawed) predicted demand rather than perfect predictions (the actual demand)
    - * the deterministic predictions of Quantile Regression Forest are should be more or less the same as those generated from Random Forest
    - * as they follow the very same tree regressor generation criterion

    '''
    # create df_full via arranging df by date, hour, quarter, zone_id
    # df_full = df_15min.sort_values(by=['date', 'Hour', 'Quarter', 'zone_id'])
    df_full['cluster_label'] = clustering_labels
    df_full['cluster_demand_actual'] = cluster_demand_actual
    df_full['cluster_demand_pred'] = cluster_demand_pred

    # calculate the error/difference between the actual and predicted demand for each cluster
    zone_wise_MAE = np.zeros(20)
    zone_wise_RMSE = np.zeros(20)
    zone_wise_RMSLE = np.zeros(20)
    zone_wise_residual = np.zeros(20)

    for i in range(20):
        zone_name = f'zone {i+1}'
        temp = df_full[df_full.OrZone == zone_name]
        mae, rmse, rmsle, resid_std = prediction_evaluate(temp.cluster_demand_pred.values, temp.cluster_demand_actual.values)
        zone_wise_MAE[i] = mae
        zone_wise_RMSE[i] = rmse
        zone_wise_RMSLE[i] = rmsle
        zone_wise_residual[i] = resid_std
    
    if print_out:
        # print a summary of the evaluation results
        print(f'MAE: {np.mean(zone_wise_MAE):.3f}({np.std(zone_wise_MAE):.3f})')
        print(f'RMSE: {np.mean(zone_wise_RMSE):.3f}({np.std(zone_wise_RMSE):.3f})')
        print(f'RMSLE: {np.mean(zone_wise_RMSLE):.3f}({np.std(zone_wise_RMSLE):.3f})')
        print(f'Residual STD: {np.mean(zone_wise_residual):.3f}({np.std(zone_wise_residual):.3f})')
    else:
        return {
            "MAE": zone_wise_MAE,
            "RMSE": zone_wise_RMSE,
            "RMSLE": zone_wise_RMSLE,
            "Residual STD": zone_wise_residual
        }