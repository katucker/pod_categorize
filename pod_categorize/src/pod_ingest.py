# -*- coding: utf-8 -*-
"""
Project Open Data inventory file ingestion

This program loads the contents of data.json files compliant with
the Project Open Data schema, cleans some of the field content, and
prepares it for analysis.
"""
import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

pod_url = 'https://www.hhs.gov/data.json'
pod = pd.read_json(pod_url)

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
        
def get_description_content(s: pd.Series) -> str:
    desc = s['description']
    # Stip any HTML tags out of the description text.
    parsed_desc = BeautifulSoup(desc,'html.parser')
    desc_text = parsed_desc.find_all(text=True)
    return desc_text[0]
    
# Extracts the first part of the publisher name (up to the first comma) and
# decodes HTML escaped ampersands.
def extract_publishers(s: pd.Series) -> str:
    return s['publisher']['name'].split(',')[0].replace('amp;','')

def tokenize_text(s: pd.Series) -> pd.Series:
    tokens = nltk.word_tokenize(s['clean_description'])
    tokens.extend(s['clean_keywords'])
    return pd.Series(tokens)
    
# Extract the column labeled dataset, transforming it from a Series
# of dictionaries into a DataFrame by applying the Series constructor
# to each dictionary.
dataset_metadata = pod['dataset'].apply(pd.Series)

# Clean the keyword text in each inventory entry.
dataset_metadata['clean_keywords'] = dataset_metadata.apply(clean_keywords, axis=1)

# Clean the description text in each inventory entry.
dataset_metadata['clean_description'] = dataset_metadata.apply(get_description_content, axis=1)

# Split the description text into a series of tokens.
dataset_metadata['tokenized_text'] = dataset_metadata.apply(tokenize_text, axis=1)

# Tag the parts of speech in each description.
tagged_description_content = pd.Series([nltk.pos_tag(dct) for dct in dataset_metadata['tokenized_text']])

dataset_metadata['publisher_name'] = dataset_metadata.apply(extract_publishers, axis=1)    

# Apply Named Entity Recognition to the description text.

# Join the keywords to the description and generate a term frequency-inverse document frequency matrix of the results
vect = TfidfVectorizer(strip_accents='unicode', ngram_range=(1,6), stop_words='english')

