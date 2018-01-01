# -*- coding: utf-8 -*-
"""
Project Open Data inventory exploration

This is an exploratory set of Python code for data.json files compliant with
the Project Open Data schema.
"""
#%%
import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import subprocess
import uuid
import os
import sklearn
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_samples

#%%
pod_url = 'https://www.hhs.gov/data.json'
pod = pd.read_json(pod_url)

# Extract the column labeled dataset, transforming it from a Series
# of dictionaries into a DataFrame by applying the Series constructor
# to each dictionary.
dataset_metadata = pod['dataset'].apply(pd.Series)

#%%
print('{} datasets have no identifier.'.format(dataset_metadata['identifier'].isnull().sum()))

print('{} of the {} datasets have unique identifiers'.format(dataset_metadata['identifier'].nunique(),dataset_metadata.shape[0]))

keywords = dataset_metadata['keyword']

# Compile a list of keywords from the datasets that only have 1.
single_keywords = [v[0] for v in keywords if len(v) == 1]

single_keywords_series = pd.Series(single_keywords)


print ('{} datasets have only 1 keyword.'.format(len(single_keywords)))
print('{} of those keywords are unique'.format(single_keywords_series.nunique()))

#%%
# Discovery: Some keyword lists are incorrectly formatted.
# A list of words with double spaces separating terms, instead of
# an array of terms.
# A list of words (some hyphenated) separated by spaces.

# Splits a string into a list of terms.
# Assumes that multiple dash characters are an encoding of words and
# replaces the dashes with spaces.
# Assumes that double spaces are a term separator.
# Assumes that several words in the string are actually a list of terms
# to separate into most-likely n-grams.
def split_terms(s: str) -> list:
    keyword_list = []
    space_encoders = r"[_-]"
    # Use a local copy of the string parameter
    working_string = s
    # Decode string by replacing dashes and underscores with spaces,
    # but only if there are more than one of them in the string.
    if (len(re.findall(space_encoders,working_string)) > 1):
        working_string = re.sub(space_encoders,' ', working_string)
    # Split the decoded string on double spaces, which is sometimes
    # used to separate multiple terms in the same string.
    working_list = working_string.split("  ")
    # TO DO: Check each string in the resulting list to see how many
    # words are in it. If the number of words is suspiciously high,
    # break the string into n-grams that are more likely to be
    # distinct terms. This corrects for keywords strings that have multiple
    # multi-word terms separated by a single space, such as:
    # "Air Quality Air Pollution"
    # which should result in:
    # ["Air Quality","Air Pollution"]
    keyword_list.extend(working_list)
    return keyword_list
    
# Scans the keyword array of the passed Series and cleanses the entries.
# This corrects keyword lists that are incorrectly formatted as a 
# single string with distinct terms separated by double spaces.
# It replaces multiple "-" or "_" characters with spaces.
def clean_keywords(s: pd.Series) -> list:
    keyword_list = []
    for kl in s['keyword']:
        keyword_list.extend(split_terms(kl))
    return keyword_list
        
dataset_metadata['clean_keywords'] = dataset_metadata.apply(clean_keywords, axis=1)

def get_description_content(s: pd.Series) -> str:
    desc = s['description']
    # Stip any HTML tags out of the description text.
    parsed_desc = BeautifulSoup(desc,'html.parser')
    desc_text = parsed_desc.find_all(text=True)
    return desc_text[0]
    
dataset_metadata['clean_description'] = dataset_metadata.apply(get_description_content, axis=1)

def tokenize_text(s: pd.Series) -> []:
    tokens = nltk.word_tokenize(s['clean_description'])
    tokens.extend(s['clean_keywords'])
    return tokens

dataset_metadata['full_text_tokens'] = dataset_metadata.apply(tokenize_text, axis=1)

def remove_stop_words(s: pd.Series) -> pd.Series:
    sig_tokens = [word for word in s['full_text_tokens'] if word not in nltk.corpus.stopwords.words('english')]
    return sig_tokens
    
# Remove the stop words from the tokens into a separate list for processing.
dataset_metadata['significant_text_tokens'] = dataset_metadata.apply(remove_stop_words, axis=1)

# Extracts the first part of the publisher name (up to the first comma) and
# decodes HTML escaped ampersands.
def extract_publishers(s: pd.Series) -> str:
    return s['publisher']['name'].split(',')[0].replace('amp;','')
    
dataset_metadata['publisher_name'] = dataset_metadata.apply(extract_publishers, axis=1)    

#%%
def get_metamap_concepts(text: str, use_disambiguation = True, identify_terms = False) -> []:
    # Path to the MetaMap application.
    mm = "/Users/ktucker/Documents/DevOps/public_mm/bin/metamap"
    # The parameter list to use for executing the MetaMap application.
    mmp = [mm, "-N"]
    if use_disambiguation:
        mmp.append("-y")
    if identify_terms:
        mmp.append("-z")
    mmproc = subprocess.Popen(mmp, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        mmout, mmerr = mmproc.communicate(input=text.encode(), timeout=120)
    except subprocess.TimeoutExpired:
        mmproc.kill()
        mmout, mmerr = mmproc.communicate()
    clist = mmout.decode().split("\n")
    # Discard the first line of output, which contains runtime parameters.
    # Discard the last line of output, since it is a blank after splitting.
    clist = clist[1:-1]
    return clist

def identify_description_concepts(s: pd.Series) -> []:
    """Identify concepts in the Unified Medical Language System found within description field of the passed Series object.
    """
    return get_metamap_concepts(s['description'])

def identify_homepage_concepts(s: pd.Series) -> []:
    """Identify concepts in the Unified Medical Language System found within the homepage text for the passed Series object.
    """
    return get_metamap_concepts(s['homepage_text'])

def identify_keyword_concepts(s: pd.Series) -> []:
    """Identify the concepts in the Unified Medical Language System within the keywords for the passed Series object.
    """
    return get_metamap_concepts(s['keywords'].join("\n"), identify_terms=True)

dataset_metadata['description_umls_concepts'] = dataset_metadata.apply(identify_description_concepts, axis=1)

#%%
def retrieve_dkan_dataset_urls(s: str) -> []:
    """Retrieves the URLS from the metadata for a dataset through the DKAN API.
    """
    dkan_metadata_url = "https://www.healthdata.gov/api/3/action/package_show?id="+s
    dkan_metadata = pd.read_json(dkan_metadata_url)
    dkan_api_results = dkan_metadata['result'].apply(pd.Series)
    dkan_urls = list(dkan_api_results['url'])
    return dkan_urls

def retrieve_dkan_urls(s: pd.Series) -> []:
    return retrieve_dkan_dataset_urls(s['identifier'])

dkan_urls = dataset_metadata['identifier'].apply(retrieve_dkan_dataset_urls)

# Plot a histogram of the publishers.
#dataset_publisher_counts = dataset_metadata['publisher_name'].value_counts()
#dataset_publisher_counts.plot(kind='barh')

def get_cwd_tempfilename():
    return os.fsencode(os.getcwd() + "/" + str(uuid.uuid4())).decode()

def get_metamap_concepts(s, disambiguate=True, identify_terms=False):
    """Retrieve a list of concepts for the passed text using the
    MetaMap application from the Unified Medical Language System.
    """
    # Path to the MetaMap application.
    mm = os.fsencode("C:/Users/Keith.Tucker/Apps/public_mm/bin/metamap14.bat").decode()
    mmp = [mm, "-N"]
    if disambiguate:
        mmp.append("-y")
    if identify_terms:
        mmp.append("-z")
    # Generate temporary file names to use for the input and output.
    # Note, this approach is used because piping stdin and stdout does
    # not work reliably on Windows systems.
    mmifn = get_cwd_tempfilename()
    mmofn = mmifn + ".out"
    mmif = open(mmifn, mode="w+t", encoding='ascii', errors='ignore')
    # If the first argument passed is an array of strings,
    # print one element per line.
    if isinstance(s, list):
        for i in s:
            print(i, file=mmif)
            print("\n", file=mmif)
    else:
        print(s, file=mmif)
        print("\n", file=mmif)
    mmif.close()
    mmp.append(mmifn)
    mmp.append(mmofn)
    try:
        subprocess.run(mmp, timeout=120, check=True)
        fmmi_names = ["score", "term", "sem", "mesh"]
        mmo = pd.read_csv(mmofn, sep="|", header=None, names=fmmi_names, usecols=[2,3,5,9])
        mmo['mesh'] = mmo['mesh'].apply(lambda x: str(x).split(";"))
#        mmof = open(mmofn, mode="r+t", encoding="ascii", errors='ignore')
#        mmo = mmof.read()
        # Break the output on newlines and discard the final empty line.
#        mmo = mmo.split("\n")[:-1]
#        mmof.close()
    except subprocess.TimeoutExpired:
        print("MetaMap run timed out.", "Input: ", s)
        mmo = []
    except subprocess.CalledProcessError:
        print("Metamap run completed with errors.", "Input: ", s)
        mmo = []
    finally:
        try:
            os.remove(mmifn)
            os.remove(mmofn)
        except os.error:
            pass
    return mmo

dataset_metadata['description_concepts'] = dataset_metadata['clean_description'].apply(get_metamap_concepts)

dataset_metadata['title_concepts'] = dataset_metadata['title'].apply(get_metamap_concepts, identify_terms=True)

dataset_metadata['keyword_concepts'] = dataset_metadata['clean_keywords'].apply(get_metamap_concepts, identify_terms=True)
#%%

# Construct the vectors to use for the clustering algorithm.

def get_dict_from_metamap(mm):
    """Transform the DataFrame passed to a dict with term names as keys and
    MMI scores as values.
    """
    mm_dict = {}
    for tr in mm.itertuples():
        try:
            val = float(tr.score)
            mm_dict[tr.term] = val
        except ValueError:
            pass
    return mm_dict


dataset_metadata['description_mm_dict'] = dataset_metadata['description_concepts'].apply(get_dict_from_metamap)

dataset_metadata['title_mm_dict'] = dataset_metadata['title_concepts'].apply(get_dict_from_metamap)

dataset_metadata['keyword_mm_dict'] = dataset_metadata['keyword_concepts'].apply(get_dict_from_metamap)

def combine_mm_dicts(s):
    """Combine the title, description and keyword MetaMap dictionaries. Keep the
    maximum score for every term found in common for a given dataset.
    """
    mmd = s['description_mm_dict']
    for tdk, tdv in s['title_mm_dict'].items():
        if tdk in mmd:
            if mmd[tdk] < tdv:
                mmd[tdk] = tdv
        else:
            mmd[tdk] = tdv
    for kdk, kdv in s['keyword_mm_dict'].items():
        if kdk in mmd:
            if mmd[kdk] < kdv:
                mmd[kdk] = kdv
        else:
            mmd[kdk] = kdv
    return mmd
            
dataset_metadata['combined_mm_dict'] = dataset_metadata.apply(combine_mm_dicts, axis=1)

mm_vect = DictVectorizer()
mm_v = mm_vect.fit_transform(list(dataset_metadata['combined_mm_dict']))

#%%
# Apply the K-means clustering algorithm on a range of cluster numbers and
# compare the resulting inertias.
# Use the same random seed for each run to avoid introducing additional
# variation to the comparison.
MAX_CLUSTERS = 50
DIMENSIONS = (7.0, 5.0)
inertias = []
for nc in range(1,MAX_CLUSTERS):
    print("Training K-Means for {} clusters.".format(nc))
    mmkm = KMeans(n_clusters = nc,
                  init = "k-means++",
                  n_init = 12,
                  max_iter = 300,
                  random_state = 1247)
    mmkm.fit(mm_v)
    inertias.append(mmkm.inertia_)
    
fig = plt.figure(figsize = DIMENSIONS)
diag = fig.add_subplot(1, 1, 1)
diag.plot(range(1, MAX_CLUSTERS), inertias, 'bo-')
diag.set_xlabel('Number clusters')
diag.set_ylabel('Inertia')
diag.set_title('Metamap concept K-Means clustering')
fig.show()
    
#%%    
# Rerun the clustering algorithm using a chosen number of clusters based on
# the plot of the inertias.
NUM_CLUSTERS = 13
mmkm = KMeans(n_clusters = NUM_CLUSTERS,
              init = 'k-means++',
              n_init = 12,
              max_iter = 300)
mm_clustering = mmkm.fit_predict(mm_v)

dataset_metadata['mmkm_cluster'] = pd.Series(mm_clustering)

#%%
# Plot the silhouettes of the MetaMap concepts against the trained clusters.
silhouettes = silhouette_samples(mm_v, mm_clustering, metric='euclidean')
average_silhouette = np.mean(silhouettes)
# Determine how many clusters actually contain values.
cluster_labels = np.unique(mm_clustering)
non_empty_clusters = cluster_labels.shape[0]
# Iterate over the non-empty cluster labels, plotting the silhouette values
# for each.
DIMENSIONS = (7.0, 5.0)
y_lower = 0
y_upper = 0
y_ticks = []
sfig, sdiags = plt.subplots(nrows = non_empty_clusters, ncols=1, sharex=True,
                            figsize = DIMENSIONS)
for d,l in enumerate(cluster_labels):
    # Get a list of silhouette values in the corresponding cluster.
    cluster = sorted(silhouettes[mm_clustering == l])
    # Increment the y-axis upper limit by the number of entries to plot.
    y_upper += len(cluster)
    sdiags[d].barh(range(0,len(cluster)), cluster, height = 1.0, 
          edgecolor = 'none', color = 'C{}'.format(l % 9))
    # Add a vertical line showing the average silhouette value.
    sdiags[d].axvline(average_silhouette, color = 'black', linestyle = '--')
    sdiags[d].set_ylabel("Cluster {}".format(l), rotation='horizontal',
          ha = 'right', va='center')
    sdiags[d].set_yticklabels([])
plt.subplots_adjust(hspace=0.0)
# Position a label at the bottom of the figure to serve as a common X-axis label.
sfig.text(0.5, 0.04, 'Silhouette coefficient', ha='center')
sfig.suptitle('Silhouettes for {} K-Means clusters'.format(len(cluster_labels)))
sfig.show()

#%%
# Apply Latent Dirichlet Allocation on the text of the inventory.
# First, create an Term Frequenecy - Inverse Document Frequency vectorization
# of the text components.
MAX_FEATURES = 1000
COMPONENTS = 20
ITERATIONS = 10
OFFSET = 10.0
tfidf_vect = TfidfVectorizer(max_df = 0.95, min_df = 2, 
                             max_features = MAX_FEATURES, 
                             stop_words = 'english')
# Combine all the cleansed text.
def concat_text_fields(s):
    retstr = s['title'] + '\n'
    for kw in s['clean_keywords']:
        retstr += kw + '\n'
    retstr += '\n' + s['clean_description']
    return retstr
dataset_text = [concat_text_fields(ds) for i,ds in dataset_metadata.iterrows()]
term_matrix = tfidf_vect.fit_transform(dataset_text)
lda = LatentDirichletAllocation(n_components = COMPONENTS,
                                max_iter = ITERATIONS,
                                learning_offset = OFFSET)
lda_clustering = lda.fit_transform(term_matrix)