#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 09:13:47 2018

@author: ktucker
"""
from flask import Flask, render_template, redirect, request
import pandas as pd
import numpy as np
import requests

SEARCH_BASE_URL = "https://ww.healthdata.gov/search/ype/dataset"

DATASET_INVENTORY_PICKLE = "..\\dataset_inventory_clustered.pickle"

CLUSTER_FIELD_NAMES = ['mm_agg_cosine_cluster',
                       'mm_agg_euclid_cluster',
                       'mm_dbscan_cosine_cluster',
                       'mm_dbscan_euclid_cluster',
                       'mm_km_cluster',
                       'tfidf_agg_cosine_cluster',
                       'tfidf_agg_euclid_cluster',
                       'tfidf_dbscan_cosine_cluster',
                       'tfidf_dbscan_euclid_cluster',
                       'tfidf_km_cluster']

CLUSTER_URLS = ["mmc-agglomerative-cosine",
                "mmc-agglomerative-euclid",
                "mmc-dbscan-cosine",
                "mmc-dbscan-euclid",
                "mmc-kmeans",
                "tf-idf-agglomerative-cosine",
                "tf-idf-agglomerative-euclid",
                "tf-idf-dbscan-cosine",
                "tf-idf-dbscan-euclid",
                "tf-idf-kmeans"]
        
CLUSTER_FILE_NAMES = ['hdg_mmc_agg_cosine_silhouettes.svg',
                      'hdg_mmc_agg_euclid_silhouettes.svg',
                      'hdg_mmc_dbscan_cosine_silhouettes.svg',
                      'hdg_mmc_dbscan_euclid_silhouettes.svg',
                      'hdg_mmc_kmeans_silhouettes.svg',
                      'hdg_tfidf_agg_cosine_silhouettes.svg',
                      'hdg_tfidf_agg_euclid_silhouettes.svg',
                      'hdg_tfidf_dbscan_cosine_silhouettes.svg',
                      'hdg_tfidf_dbscan_euclid_silhouettes.svg',
                      'hdg_tfidf_kmeans_silhouettes.svg']

CLUSTER_TITLES = ['Agglomerative Clustering of MetaMap concepts using Cosine distance metric.',
                  'Agglomerative Clustering of MetaMap concepts using Euclidean distance metric.',
                  'DBSCAN Clustering of MetaMap concepts using Cosine distance metric.',
                  'DBSCAN Clustering of MetaMap concepts using Euclidean distance metric.',
                  'K-Means Clustering of MetaMap concepts.',
                  'Agglomerative Clustering of term frequency using Cosine distance metric.',
                  'Agglomerative Clustering of term frequency using Euclidean distance metric.',
                  'DBSCAN Clustering of term frequency using Cosine distance metric.',
                  'DBSCAN Clustering of term frequency using Euclidean distance metric.',
                  'K-Means Clustering of term frequency.']

# Read the dataset inventory Dataframe.
dataset_inventory = pd.read_pickle(DATASET_INVENTORY_PICKLE)

# Create a Flask app instance.
app = Flask(__name__)

def render_cluster(cluster_number = 0, cluster_index = 0):
    # Force the cluster_number to within range.
    if cluster_number < 0:
        cluster_number = 0
    elif cluster_number >= len(CLUSTER_FIELD_NAMES):
        cluster_number = len(CLUSTER_FIELD_NAMES) -1
    cluster_name = CLUSTER_FIELD_NAMES[cluster_number]
    cluster_url = CLUSTER_URLS[cluster_number]
    cluster_indices = dataset_inventory[cluster_name]
    clusters = list(np.unique(cluster_indices))
    cluster_title = CLUSTER_TITLES[cluster_number]
    diagram = CLUSTER_FILE_NAMES[cluster_number]
    # Force the cluster_index to within range.
    if cluster_index < 0:
        cluster_index = 0
    elif cluster_index >= len(clusters):
        cluster_index = len(clusters)-1
    cluster_dataset_titles = []
    for index, dsr in dataset_inventory.iterrows():
        if dsr[cluster_name] == clusters[cluster_index]:
            cluster_dataset_titles.append((index, dsr.title))
    return render_template('cluster_list.html',
                           cluster_url = cluster_url,
                           cluster_indices = cluster_indices,
                           clusters = clusters,
                           cluster_title = cluster_title,
                           cluster_index = cluster_index,
                           cluster_dataset_titles = cluster_dataset_titles,
                           diagram = diagram)
    
# Define the default route as a wrapper search page.
@app.route('/')
def show_default_clustering():
    return render_cluster()

@app.route('/mmc-agglomerative-cosine/<int:cluster_index>')
def display_mmc_agg_cosine(cluster_index = 0):
    return render_cluster(0, cluster_index)
    
@app.route('/mmc-agglomerative-euclid/<int:cluster_index>')
def display_mmc_agg_euclid(cluster_index = 0):
    return render_cluster(1, cluster_index)

@app.route('/mmc-dbscan-cosine/<int:cluster_index>')
def display_mmc_dbscan_cosine(cluster_index = 0):
    return render_cluster(2, cluster_index)

@app.route('/mmc-dbscan-euclid/<int:cluster_index>')
def display_mmc_dbscan_euclid(cluster_index = 0):
    return render_cluster(3, cluster_index)

@app.route('/mmc-kmeans/<int:cluster_index>')
def display_mmc_kmeans(cluster_index = 0):
    return render_cluster(4, cluster_index)

@app.route('/tf-idf-agglomerative-cosine/<int:cluster_index>')
def display_tfidf_agg_cosine(cluster_index = 0):
    return render_cluster(5, cluster_index)
    
@app.route('/tf-idf-agglomerative-euclid/<int:cluster_index>')
def display_tfidf_agg_euclid(cluster_index = 0):
    return render_cluster(6, cluster_index)

@app.route('/tf-idf-dbscan-cosine/<int:cluster_index>')
def display_tfidf_dbscan_cosine(cluster_index = 0):
    return render_cluster(7, cluster_index)

@app.route('/tf-idf-dbscan-euclid/<int:cluster_index>')
def display_tfidf_dbscan_euclid(cluster_index = 0):
    return render_cluster(8, cluster_index)

@app.route('/tf-idf-kmeans/<int:cluster_index>')
def display_tfidf_kmeans(cluster_index = 0):
    return render_cluster(9, cluster_index)

@app.route('/dataset/<int:index>')
def display_dataset(index = 0):
    # Force the index to be in range.
    if index < 0:
        index = 0
    elif index >= len(dataset_inventory):
        index = len(dataset_inventory)-1
    return render_template('dataset_view.html',
                           dataset_title = dataset_inventory.iloc[index].title,
                           dataset_description = dataset_inventory.iloc[index].clean_description,
                           dataset_keywords = dataset_inventory.iloc[index].clean_keywords,
                           dataset_metamap_concepts = dataset_inventory.iloc[index].combined_mm_dict)

app.run(port=5000)

