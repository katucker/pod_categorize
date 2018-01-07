#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 09:13:47 2018

@author: ktucker
"""
from flask import Flask, render_template, redirect, request
import pandas as pd
import requests

SEARCH_BASE_URL = "https://ww.healthdata.gov/search/ype/dataset"
DATASET_INVENTORY_PICKLE = "dataset_inventory"

# Read the dataset inventory Dataframe.
dataset_inventory = pd.read_pickle(DATASET_INVENTORY_PICKLE)

# Create a Flask app instance.
flask_app = Flask(__name__)

# Define the default route as a wrapper search page.
@flask_app.route('/')
def show_clustering_choices():
    return render_template('cluster_view.html')

@flask_app.route('/search_ml')
def augmented_search(search_term):
    hd_url = SEARCH_BASE_URL + request.form['search']
    response_html = requests.post(url=SEARCH_BASE_URL, data=request.form)
    parsed_response = BeautifulSoup(response_html, 'html.parser')
