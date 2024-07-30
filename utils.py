import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
pd.set_option('display.max_columns', None)

import re
import numpy as np
import random
from random import randint
import time

import fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import rapidfuzz
from joblib import Parallel, delayed
import multiprocessing


def preprocess_name(name):
    name = name.lower()
    name = re.sub(r'[^a-z\s]', '', name)  # Remove special characters
    name = re.sub(r'\bjr\b|\bsr\b|\biii\b', '', name)  # Remove common suffixes
    return name.strip()

# FuzzyWuzzy matching (extracting top1 match for each row/combination)
def fuzzy_match_with_score(row, choices, scorer, threshold=80):
    choices_list = list(choices)  # Ensure choices is a list
    match = fuzzywuzzy.process.extractOne(row, choices_list, scorer=scorer)
    if match and match[1] >= threshold:
        matched_value, score = match
        match_index = choices_list.index(matched_value)
        return matched_value, score, match_index
    return None, 0, -1

# RapidFuzz matching (extracting top1 match for each row/combination)
def rapidfuzz_match_with_score(row, choices, scorer, threshold=80):
    choices_list = list(choices)  # Ensure choices is a list
    match = process.extractOne(row, choices_list, scorer=scorer)
    if match and match[1] >= threshold:
        matched_value, score, match_index = match[0], match[1], choices_list.index(match[0])
        return matched_value, score, match_index
    return None, 0, -1

# Optimized function to get the best match using cdist
def optimized_rapidfuzz_match_with_score(df_col, choices, scorer):
    results = rapidfuzz.process.cdist(queries=df_col.dropna().tolist(), choices=choices, scorer=scorer)
    best_matches = results.argmax(axis=1)
    best_scores = results.max(axis=1)
    matched_values = [choices[idx] for idx in best_matches]

    # Create a DataFrame to hold the results
    matched_df = pd.DataFrame({
        'Matched': matched_values,
        'Score': best_scores,
        'Index': best_matches
    }, index=df_col.dropna().index)

    # Reindex to the original DataFrame's index
    matched_df = matched_df.reindex(df_col.index, fill_value=None)

    return matched_df['Matched'], matched_df['Score'], matched_df['Index']

def parallel_rapidfuzz_match_with_score(df_col, choices, scorer, n_jobs):
    # Split the dataframe column into chunks
    chunk_size = len(df_col) // n_jobs
    chunks = [df_col[i:i + chunk_size] for i in range(0, len(df_col), chunk_size)]

    # Parallel processing using joblib
    results = Parallel(n_jobs=n_jobs)(delayed(optimized_rapidfuzz_match_with_score)(chunk, choices, scorer) for chunk in chunks)

    # Concatenate the results
    matched = pd.concat([result[0] for result in results])
    scores = pd.concat([result[1] for result in results])
    indices = pd.concat([result[2] for result in results])

    return matched, scores, indices
