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
import subprocess
import uuid
import os

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

# Plot a histogram of the publishers.
dataset_publisher_counts = dataset_metadata['publisher_name'].value_counts()
dataset_publisher_counts.plot(kind='barh')

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
    mm_dict = {tr.term: tr.score for tr in mm.itertuples()}
    return mm_dict

#%%    
# Collect the keyword and description length for each dataset.
#def calculate_keyword_count(s: pd.Series) -> {}:
#    return len(s['clean_keywords'])
#    
#def calculate_keyword_characters(s: pd.Series) -> {}:
#    return pd.Series((len(k) for k in s['clean_keywords'])).sum()
#
#def calculate_description_word_count(s: pd.Series) -> {}:
#    return len(nltk.word_tokenize(s['clean_description']))
#
#def calculate_description_characters(s: pd.Series) -> {}:
#    return len(s['clean_description'])
#
#dataset_metadata['keyword_count'] = dataset_metadata.apply(calculate_keyword_count, axis=1)
#dataset_metadata['keyword_characters'] = dataset_metadata.apply(calculate_keyword_characters, axis=1)
#dataset_metadata['description_word_count'] = dataset_metadata.apply(calculate_description_word_count, axis=1)
#dataset_metadata['description_characters'] = dataset_metadata.apply(calculate_description_characters, axis=1)
#
#dataset_metadata_by_publisher = dataset_metadata.groupby('publisher_name')
#
#for p,ilist in dataset_metadata_by_publisher:
#    fig = plt.figure()
#    diag = fig.add_subplot(1,1,1)
#    ilist.hist(column='keyword_characters', ax=diag)
#    diag.set_title(p + ' keyword character count')
#    fig.show()
#    fig2 = plt.figure()
#    diag2 = fig2.add_subplot(1,1,1)
#    ilist.hist(column='description_characters', ax=diag2)
#    diag2.set_title(p + ' description character count')
#    fig2.show()

#uts = UMLSService(args.apikey)
#apikey = input('Enter the API key to use UMLS Terminology Service: ')
#uts = UMLSService(apikey)
#print(uts.search(dataset_metadata.iloc[0]['tokenized_keywords'][0]))