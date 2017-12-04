# -*- coding: utf-8 -*-
"""
Project Open Data inventory exploration

This is an exploratory set of Python code for data.json files compliant with
the Project Open Data schema.
"""
import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt


pod_url = 'https://www.hhs.gov/data.json'
pod = pd.read_json(pod_url)

# Extract the column labeled dataset, transforming it from a Series
# of dictionaries into a DataFrame by applying the Series constructor
# to each dictionary.
dataset_metadata = pod['dataset'].apply(pd.Series)

print('{} datasets have no identifier.'.format(dataset_metadata['identifier'].isnull().sum()))

print('{} of the {} datasets have unique identifiers'.format(dataset_metadata['identifier'].nunique(),dataset_metadata.shape[0]))

keywords = dataset_metadata['keyword']

# Compile a list of keywords from the datasets that only have 1.
single_keywords = [v[0] for v in keywords if len(v) == 1]

single_keywords_series = pd.Series(single_keywords)


print ('{} datasets have only 1 keyword.'.format(len(single_keywords)))
print('{} of those keywords are unique'.format(single_keywords_series.nunique()))

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
def clean_keywords(s: pd.core.series.Series) -> list:
    keyword_list = []
    for kl in s['keyword']:
        keyword_list.extend(split_terms(kl))
    return keyword_list
        
dataset_metadata['clean_keywords'] = dataset_metadata.apply(clean_keywords, axis=1)

def get_description_content(s: pd.core.series.Series) -> str:
    desc = s['description']
    # Stip any HTML tags out of the description text.
    parsed_desc = BeautifulSoup(desc,'html.parser')
    desc_text = parsed_desc.find_all(text=True)
    return desc_text[0]
    
dataset_metadata['clean_description'] = dataset_metadata.apply(get_description_content, axis=1)

# Extracts the first part of the publisher name (up to the first comma) and
# decodes HTML escaped ampersands.
def extract_publishers(s: pd.core.series.Series) -> str:
    return s['publisher']['name'].split(',')[0].replace('amp;','')
    
dataset_metadata['publisher_name'] = dataset_metadata.apply(extract_publishers, axis=1)    

# Plot a historgam of the publishers.
dataset_publisher_counts = dataset_metadata['publisher_name'].value_counts()
dataset_publisher_counts.plot(kind='barh')

# Collect the keyword and description length for each dataset.
def calculate_keyword_count(s: pd.core.series.Series) -> {}:
    return len(s['clean_keywords'])
    
def calculate_keyword_characters(s: pd.core.series.Series) -> {}:
    return pd.Series((len(k) for k in s['clean_keywords'])).sum()

def calculate_description_word_count(s: pd.core.series.Series) -> {}:
    return len(nltk.word_tokenize(s['clean_description']))

def calculate_description_characters(s: pd.core.series.Series) -> {}:
    return len(s['clean_description'])

dataset_metadata['keyword_count'] = dataset_metadata.apply(calculate_keyword_count, axis=1)
dataset_metadata['keyword_characters'] = dataset_metadata.apply(calculate_keyword_characters, axis=1)
dataset_metadata['description_word_count'] = dataset_metadata.apply(calculate_description_word_count, axis=1)
dataset_metadata['description_characters'] = dataset_metadata.apply(calculate_description_characters, axis=1)

dataset_metadata_by_publisher = dataset_metadata.groupby('publisher_name')

for p,ilist in dataset_metadata_by_publisher:
    fig = plt.figure()
    diag = fig.add_subplot(1,1,1)
    ilist.hist(column='keyword_characters', ax=diag)
    diag.set_title(p + ' keyword character count')
    fig.show()
    fig2 = plt.figure()
    diag2 = fig2.add_subplot(1,1,1)
    ilist.hist(column='description_characters', ax=diag2)
    diag2.set_title(p + ' description character count')
    fig2.show()
