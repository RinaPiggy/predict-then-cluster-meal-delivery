# import packages
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import holidays

# clustering packages

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
# a k-means algorithm which enable the constrain of minimum number of members per cluster
from k_means_constrained import KMeansConstrained
# Documentation -> https://joshlk.github.io/k-means-constrained/#:~:text=one%20are%20used.-,Notes,%2Dscaling%20push%2Drelabel%20algorithm.

# visualization packages
import h3
from geojson.feature import *
from folium import Map, Marker, GeoJson
from folium.plugins import MarkerCluster
import branca.colormap as cm
from branca.colormap import linear
import folium
import json
import plotly.graph_objects as go
import io
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from joblib import dump, load
import joblib
from sklearn.model_selection import RandomizedSearchCV

zone_dict = {'zone 1': (52.380706833707706, 4.843796664723742), 'zone 2': (52.38451144444716, 4.889609349787249), 'zone 3': (52.35645423301056, 4.917352878926178), 
            'zone 4': (52.355692596673464, 4.929764121137557), 'zone 5': (52.37313291102659, 4.860043977261975), 'zone 6': (52.37162230263845, 4.884865432622304), 
            'zone 7': (52.35341822482602, 4.859143202161826), 'zone 8': (52.38526726951066, 4.877195481815542), 'zone 9': (52.36175760576193, 4.938346953254661), 
            'zone 10': (52.37086494272505, 4.897277258477512), 'zone 11': (52.35721449838103, 4.904942365291374), 'zone 12': (52.37768868645137, 4.89344365444578), 
            'zone 13': (52.35114745992647, 4.8963651052972565), 'zone 14': (52.35948707135332, 4.867715206847425), 'zone 15': (52.36631048401517, 4.863879942292339), 
            'zone 16': (52.36251984616686, 4.925933961187669), 'zone 17': (52.37237829198337, 4.872454338517145), 'zone 18': (52.365555097448286, 4.876289283545159), 
            'zone 19': (52.35797339305077, 4.8925325813334615), 'zone 20': (52.364798340646516, 4.888699357161011)}

# Modules for dynamic clustering 
def data_normalization(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features

def cluster(k=3, cluster_method ='constrained_k-means'):
    if cluster_method =='k-means++':
        kmeans = KMeans(n_clusters= k, 
                        init = 'k-means++',
                        random_state=66)

    elif cluster_method =='constrained_k-means':
        kmeans = KMeansConstrained(
                        n_clusters=k,
                        size_min=3,
                        init = 'k-means++',
                        random_state=66)
    return kmeans

def clustering(scaled_features,min_k=3, max_k=6, cluster_method='constrained_k-means'):
    sse = []
    avg_silhouettes = []
    # run through all the candidate k
    for k in range(min_k,max_k+1):
        kmeans = cluster(k,cluster_method)
        cluster_labels = kmeans.fit_predict(scaled_features)
        sse.append(kmeans.inertia_)
        avg_silhouettes.append(silhouette_score(scaled_features, cluster_labels))
    
    # taking the number of clusters which corresponds to the highest avg_silhouettes
    id = avg_silhouettes.index(max(avg_silhouettes))
    opt_k = range(min_k,max_k+1)[id]
    new_kmeans = cluster(opt_k,cluster_method)
    cluster_labels = new_kmeans.fit_predict(scaled_features)

    return cluster_labels, new_kmeans.cluster_centers_, opt_k

def stepwise_clustering_over_real(df_test,test_set_size=308,method='constrained_k-means'):

    # label "prediction" actually refers to the actual demand, but for convienience we keep the previous name 
    features = ['prediction','latitude','longitude']

    optimal_ks = []
    real_median_per_cluster = []
    real_mean_per_cluster = []

    for i in tqdm(range(test_set_size)):
        real_dict = pd.DataFrame(columns=features)
        for z in list(zone_dict.keys())[:20]:
            address = zone_dict[z]
            # temp = all_test_y_with_hour[z].iloc[i]
            temp = df_test[df_test['OrZone']==z].iloc[i]
            # record the prediction information [prediction, address]
            real_dict = real_dict.append({'prediction':temp['counts'],'latitude':address[0],'longitude':address[1]},ignore_index=True)
        
        # collect the predictions, and create a clustering heatmap at each step
        labels, _, temp2 = clustering(data_normalization(real_dict),min_k=3, max_k=6, cluster_method=method)
        optimal_ks.append(temp2)
        real_dict['k-label'] = labels
        median_df = real_dict.groupby('k-label')['prediction']
        real_dict['real_cluster_median'] = median_df.transform('median')
        real_dict['real_cluster_mean'] = median_df.transform('mean')
        real_median_per_cluster = real_median_per_cluster + real_dict['real_cluster_median'].tolist()
        real_mean_per_cluster = real_mean_per_cluster + real_dict['real_cluster_mean'].tolist()

    return real_median_per_cluster,real_mean_per_cluster,optimal_ks


# # make predictions in a rolling window style
# def stepwise_prediction_and_clustering(zone_dict,model_dict,df_test,model_name,test_set_size=308,
#                                         method='constrained_k-means'):
#     if model_name == 'QRF' or model_name == 'ARQRF':
#         # features = ['p25_prediction','prediction','p75_prediction','latitude','longitude']
#         features = ['prediction','latitude','longitude']
#         features_type = 'quantiles_pred'
#     else:
#         features = ['prediction','latitude','longitude']
#         features_type = 'single_pred'

#     optimal_ks = []
#     all_pred_labels = []
#     pred_median_per_cluster = []
#     pred_mean_per_cluster = []

#     for i in tqdm(range(test_set_size)):
#         # create a dataframe of all the prediction information at the time window i
#         pred_dict = pd.DataFrame(columns=features)
#         pred_median = np.zeros(20)
#         for j,z in enumerate(list(zone_dict.keys())[:20]):
#             model = model_dict[z]
#             df_zone = df_test[df_test['OrZone']==z]
#             X = df_zone.iloc[i:i+1,:]
#             if model_name == 'ARRF' or model_name == 'ARQRF' or model_name == 'ARXGB':
#                 X = X[['DayofWeek','Hour','temp','wspd','prep','Is_Holiday','AR1','AR2','AR3','AR4']]
#             else:
#                 X = X[['DayofWeek','Hour','temp','wspd','prep','Is_Holiday']]
#             address = zone_dict[z]
#             # record the prediction information [prediction, address]
#             if features_type == 'single_pred':
#                 if model_name == 'TBATS':
#                     y_all = model.forecast(steps=(i+1))
#                     y50 = [y_all[-1]]
#                 else:
#                     y50 = model.predict(X)
#                 pred_dict = pred_dict.append({'prediction':y50[0],'latitude':address[0],'longitude':address[1]},ignore_index=True)
                
#             elif features_type == 'quantiles_pred':
#                 y75 = model.predict(X.to_numpy(), quantile=75)
#                 y50 = model.predict(X.to_numpy(), quantile=50)
#                 y25 = model.predict(X.to_numpy(), quantile=25)
#                 # TODO: NOTE I MADE CHANGES HERE
#                 y = 1*y75[0] + 2* y50[0] +1*y25[0]
#                 pred_dict = pred_dict.append({'prediction':y,'latitude':address[0],'longitude':address[1]},ignore_index=True)
#                 pred_median[j] = y50
#         # collect the predictions, and create a clustering heatmap at each step
#         norm_matrix = data_normalization(pred_dict)
#         labels, _, temp2 = clustering(data_normalization(pred_dict),min_k=3, max_k=6, cluster_method=method)
#         optimal_ks.append(temp2)
#         pred_dict['k-label'] = labels
#         if features_type:
#             pred_dict['prediction_'] = pred_median
#         all_pred_labels = all_pred_labels + pred_dict['k-label'].tolist()
#         # calculate the predicted median demand per cluster given by kmeans, groupby on pred_dict first 
#         median_df = pred_dict.groupby('k-label')['prediction_']
#         pred_dict['pred_cluster_median'] = median_df.transform('median')
#         pred_dict['pred_cluster_mean'] = median_df.transform('mean')
#         pred_median_per_cluster = pred_median_per_cluster + pred_dict['pred_cluster_median'].tolist()
#         pred_mean_per_cluster = pred_mean_per_cluster + pred_dict['pred_cluster_mean'].tolist()

#     return pred_median_per_cluster,pred_mean_per_cluster,optimal_ks


# def stepwise_clustering_over_real(df_test,test_set_size=308,method='constrained_k-means'):

#     # label "prediction" actually refers to the actual demand, but for convienience we keep the previous name 
#     features = ['prediction','latitude','longitude']

#     optimal_ks = []
#     real_median_per_cluster = []
#     real_mean_per_cluster = []

#     for i in tqdm(range(test_set_size)):
#         real_dict = pd.DataFrame(columns=features)
#         for z in list(zone_dict.keys())[:20]:
#             address = zone_dict[z]
#             # temp = all_test_y_with_hour[z].iloc[i]
#             temp = df_test[df_test['OrZone']==z].iloc[i]
#             # record the prediction information [prediction, address]
#             real_dict = real_dict.append({'prediction':temp['counts'],'latitude':address[0],'longitude':address[1]},ignore_index=True)
        
#         # collect the predictions, and create a clustering heatmap at each step
#         labels, _, temp2 = clustering(data_normalization(real_dict),min_k=3, max_k=6, cluster_method=method)
#         optimal_ks.append(temp2)
#         real_dict['k-label'] = labels
#         median_df = real_dict.groupby('k-label')['prediction']
#         real_dict['real_cluster_median'] = median_df.transform('median')
#         real_dict['real_cluster_mean'] = median_df.transform('mean')
#         real_median_per_cluster = real_median_per_cluster + real_dict['real_cluster_median'].tolist()
#         real_mean_per_cluster = real_mean_per_cluster + real_dict['real_cluster_mean'].tolist()

#     return real_median_per_cluster,real_mean_per_cluster,optimal_ks
