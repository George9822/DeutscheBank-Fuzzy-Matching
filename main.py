from utils import *

def best_match_with_score(param_df, scorer, matching_function) -> pd.DataFrame:
    # Apply fuzzy matching to df1
    df = param_df.copy()
    if (matching_function != optimized_rapidfuzz_match_with_score and matching_function!=parallel_rapidfuzz_match_with_score):
      df['MatchedFullName'], df['FullNameScore'], df['FullNameIndex'] = zip(*df['FullName'].dropna().apply(lambda x: matching_function(x, client_df_full_names, scorer=scorer)))
      df['MatchedFirstName'], df['FirstNameScore'], df['FirstNameIndex'] = zip(*df['FirstName'].dropna().apply(lambda x: matching_function(x, client_df_first_names, scorer=scorer)))
      df['MatchedLastName'], df['LastNameScore'], df['LastNameIndex'] = zip(*df['LastName'].dropna().apply(lambda x: matching_function(x, client_df_last_names, scorer=scorer)))
    elif matching_function==optimized_rapidfuzz_match_with_score:
      df['MatchedFullName'], df['FullNameScore'], df['FullNameIndex'] = optimized_rapidfuzz_match_with_score(df['FullName'], client_df_full_names, scorer)
      df['MatchedFirstName'], df['FirstNameScore'], df['FirstNameIndex'] = optimized_rapidfuzz_match_with_score(df['FirstName'], client_df_first_names, scorer)
      df['MatchedLastName'], df['LastNameScore'], df['LastNameIndex'] = optimized_rapidfuzz_match_with_score(df['LastName'], client_df_last_names, scorer)
    elif matching_function==parallel_rapidfuzz_match_with_score:
      n_jobs = multiprocessing.cpu_count()
      df['MatchedFullName'], df['FullNameScore'], df['FullNameIndex'] = parallel_rapidfuzz_match_with_score(df['FullName'], client_df_full_names, scorer, n_jobs)
      df['MatchedFirstName'], df['FirstNameScore'], df['FirstNameIndex'] = parallel_rapidfuzz_match_with_score(df['FirstName'], client_df_first_names, scorer, n_jobs)
      df['MatchedLastName'], df['LastNameScore'], df['LastNameIndex'] = parallel_rapidfuzz_match_with_score(df['LastName'], client_df_last_names, scorer, n_jobs)

    # Fill NaN values for rows that were originally NaN
    df['MatchedFullName'].fillna('_', inplace=True)
    df['FullNameScore'].fillna(0, inplace=True)
    df['FullNameIndex'].fillna(-1, inplace=True)
    df['MatchedFirstName'].fillna('_', inplace=True)
    df['FirstNameScore'].fillna(0, inplace=True)
    df['FirstNameIndex'].fillna(-1, inplace=True)
    df['MatchedLastName'].fillna('_', inplace=True)
    df['LastNameScore'].fillna(0, inplace=True)
    df['LastNameIndex'].fillna(-1, inplace=True)

    return df


# INGESTING CSV DATAFRAMES
df1 = pd.read_csv(r"C:\Users\dalia\Desktop\Radu\DeutscheBank Fuzzy Matching\dataframes\hr_df1.csv")
df2 = pd.read_csv(r"C:\Users\dalia\Desktop\Radu\DeutscheBank Fuzzy Matching\dataframes\client_df1.csv")
df1 = df1.drop(columns = ['Unnamed: 0'])
df2 = df2.drop(columns = ['Unnamed: 0'])

# Preprocessing data and drop rows with NaN values only for specific columns in both dataframes
columns_to_check = ['FirstName', 'LastName']
# df1_cleaned = df1.dropna(subset=columns_to_check)
df2 = df2.dropna(subset=columns_to_check)
df1['FirstName'] = df1['FirstName'].apply(preprocess_name)
df1['LastName'] = df1['LastName'].apply(preprocess_name)
df2['FirstName'] = df2['FirstName'].apply(preprocess_name)
df2['LastName'] = df2['LastName'].apply(preprocess_name)

print("After dropping rows with NaNs in 'FirstName' and 'LastName':")
# print("df1_cleaned:")
# print(df1_cleaned.isnull().sum())
print("df2 cleaned(no of nulls on first/last names):")
print(df2.isnull().sum())

hr_df = df1.copy()
client_df = df2.copy()

client_df['Revenue'] = [randint(2000, 500000) for _ in range(df2.shape[0])]
print(f"DF Shapes: hr_df = {hr_df.shape}, client_df = {client_df.shape}")
hr_df['FullName'] = hr_df['FirstName']+hr_df['LastName']

# Transforming the names into lowercase
client_df['FullName'] = client_df['FirstName']+client_df['LastName']

# Prepare lists of names/choices from df2 to match against
client_df_first_names = client_df['FirstName'].dropna().unique()
client_df_last_names = client_df['LastName'].dropna().unique()
client_df_full_names = client_df['FullName'].dropna().unique()

# FUZZYWUZZY - SIMPLE RATIO METHOD
start_time = time.time()
df_fuzzy_simple_ratio = best_match_with_score(param_df = hr_df, scorer = fuzz.ratio, matching_function=fuzzy_match_with_score)
print(f"Elapsed time fuzzy simple ratio: {(time.time() - start_time):.2f} seconds")
# Show results
# print(df_fuzzy_simple_ratio[['FirstName', 'MatchedFirstName', 'FirstNameScore', 'FirstNameIndex', 'LastName', 'MatchedLastName','LastNameScore','LastNameIndex','MatchedFullName','FullNameScore','FullNameIndex']].head(100))

# FUZZYWUZZY - PARTIAL RATIO METHOD
start_time = time.time()
df_fuzzy_partial_ratio = best_match_with_score(param_df = hr_df, scorer = fuzz.partial_ratio, matching_function=fuzzy_match_with_score)
print(f"Elapsed time fuzzy partial ratio: {(time.time() - start_time):.2f} seconds")
# Show results
# df_partial_ratio[['FirstName', 'MatchedFirstName', 'FirstNameScore', 'LastName', 'MatchedLastName', 'LastNameScore', 'MatchedFullName','FullNameScore']].head(100)

# FUZZYWUZZY - TOKEN_SORT_RATIO METHOD
start_time = time.time()
df_fuzzy_token_sort_ratio = best_match_with_score(param_df = hr_df, scorer = fuzz.token_sort_ratio, matching_function=fuzzy_match_with_score)
print(f"Elapsed time fuzzy token sort ratio: {(time.time() - start_time):.2f} seconds")
# Show results
# df_fuzzy_token_sort_ratio[['FirstName', 'MatchedFirstName', 'FirstNameScore', 'LastName', 'MatchedLastName', 'LastNameScore', 'MatchedFullName','FullNameScore']].head(100)

# FUZZYWUZZY - TOKEN_SET_RATIO METHOD
start_time = time.time()
df_fuzzy_token_set_ratio = best_match_with_score(param_df = hr_df, scorer = fuzz.token_set_ratio, matching_function=fuzzy_match_with_score)
print(f"Elapsed time fuzzy token set ratio: {(time.time() - start_time):.2f} seconds")
# Show results
# df_fuzzy_token_set_ratio[['FirstName', 'MatchedFirstName', 'FirstNameScore', 'LastName', 'MatchedLastName', 'LastNameScore', 'MatchedFullName','FullNameScore']].head(100)


# RAPIDFUZZ - SIMPLE RATIO METHOD
start_time = time.time()
df_rapidfuzz_simple_ratio = best_match_with_score(param_df = hr_df, scorer = rapidfuzz.fuzz.ratio, matching_function=rapidfuzz_match_with_score)
print(f"Elapsed time rapidfuzz simple ratio: {(time.time() - start_time):.2f} seconds")
# Show results
# df_rapidfuzz_simple_ratio[['FirstName', 'MatchedFirstName', 'FirstNameScore', 'FirstNameIndex', 'LastName', 'MatchedLastName','LastNameScore','LastNameIndex','MatchedFullName','FullNameScore','FullNameIndex']].head(100)

# RAPIDFUZZ - PARTIAL RATIO METHOD
start_time = time.time()
df_rapidfuzz_partial_ratio = best_match_with_score(param_df = hr_df, scorer = rapidfuzz.fuzz.partial_ratio, matching_function=rapidfuzz_match_with_score)
print(f"Elapsed time rapidfuzz partial ratio: {(time.time() - start_time):.2f} seconds")
# Show results
# df_rapidfuzz_partial_ratio[['FirstName', 'MatchedFirstName', 'FirstNameScore', 'FirstNameIndex', 'LastName', 'MatchedLastName','LastNameScore','LastNameIndex','MatchedFullName','FullNameScore','FullNameIndex']].head(100)

# RAPIDFUZZ - TOKEN SORT RATIO METHOD
start_time = time.time()
df_rapidfuzz_token_sort_ratio = best_match_with_score(param_df = hr_df, scorer = rapidfuzz.fuzz.token_sort_ratio, matching_function=rapidfuzz_match_with_score)
print(f"Elapsed time rapidfuzz token sort ratio: {(time.time() - start_time):.2f} seconds")
# Show results
# df_rapidfuzz_token_sort_ratio[['FirstName', 'MatchedFirstName', 'FirstNameScore', 'FirstNameIndex', 'LastName', 'MatchedLastName','LastNameScore','LastNameIndex','MatchedFullName','FullNameScore','FullNameIndex']].head(100)

# RAPIDFUZZ -  TOKEN SET RATIO METHOD
start_time = time.time()
df_rapidfuzz_token_set_ratio = best_match_with_score(param_df = hr_df, scorer = rapidfuzz.fuzz.token_set_ratio, matching_function=rapidfuzz_match_with_score)
print(f"Elapsed time rapidfuzz token set ratio: {(time.time() - start_time):.2f} seconds")
# Show results
# df_rapidfuzz_token_set_ratio[['FirstName', 'MatchedFirstName', 'FirstNameScore', 'FirstNameIndex', 'LastName', 'MatchedLastName','LastNameScore','LastNameIndex','MatchedFullName','FullNameScore','FullNameIndex']].head(100)

# OPTIMIZED RAPIDFUZZ - SIMPLE RATIO METHOD
start_time = time.time()
df_opt_rapidfuzz_simple_ratio = best_match_with_score(param_df = hr_df, scorer = rapidfuzz.fuzz.ratio, matching_function=optimized_rapidfuzz_match_with_score)
print(f"Elapsed time optimized rapidfuzz simple ratio: {(time.time() - start_time):.2f} seconds")
# Show results
# print(df_opt_rapidfuzz_simple_ratio[['FirstName', 'MatchedFirstName', 'FirstNameScore', 'FirstNameIndex', 'LastName', 'MatchedLastName','LastNameScore','LastNameIndex','MatchedFullName','FullNameScore','FullNameIndex']].head(100))

# OPTIMIZED RAPIDFUZZ - PARITAL RATIO METHOD
start_time = time.time()
df_opt_rapidfuzz_partial_ratio = best_match_with_score(param_df = hr_df, scorer = rapidfuzz.fuzz.partial_ratio, matching_function=optimized_rapidfuzz_match_with_score)
print(f"Elapsed time optimized rapidfuzz partial ratio: {(time.time() - start_time):.2f} seconds")
# Show results
# df_opt_rapidfuzz_[['FirstName', 'MatchedFirstName', 'FirstNameScore', 'FirstNameIndex', 'LastName', 'MatchedLastName','LastNameScore','LastNameIndex','MatchedFullName','FullNameScore','FullNameIndex']].head(100)

# OPTIMIZED RAPIDFUZZ -  TOKEN SORT RATIO METHOD
start_time = time.time()
df_opt_rapidfuzz_token_sort_ratio = best_match_with_score(param_df = hr_df, scorer = rapidfuzz.fuzz.token_sort_ratio, matching_function=optimized_rapidfuzz_match_with_score)
print(f"Elapsed time optimized rapidfuzz token sort ratio: {(time.time() - start_time):.2f} seconds")
# Show results
# df_opt_rapidfuzz_token_sort_ratio[['FirstName', 'MatchedFirstName', 'FirstNameScore', 'FirstNameIndex', 'LastName', 'MatchedLastName','LastNameScore','LastNameIndex','MatchedFullName','FullNameScore','FullNameIndex']].head(100)

# OPTIMIZED RAPIDFUZZ -  TOKEN SET RATIO METHOD
start_time = time.time()
df_opt_rapidfuzz_token_set_ratio = best_match_with_score(param_df = hr_df, scorer = rapidfuzz.fuzz.token_set_ratio, matching_function=optimized_rapidfuzz_match_with_score)
print(f"Elapsed time optimized rapidfuzz token set ratio: {(time.time() - start_time):.2f} seconds")
# Show results
# df_opt_rapidfuzz_token_set_ratio[['FirstName', 'MatchedFirstName', 'FirstNameScore', 'FirstNameIndex', 'LastName', 'MatchedLastName','LastNameScore','LastNameIndex','MatchedFullName','FullNameScore','FullNameIndex']].head(100)

# PARALLEL OPTIMIZED RAPIDFUZZ - SIMPLE RATIO METHOD
start_time = time.time()
df_parallelopt_rapidfuzz_simple_ratio = best_match_with_score(param_df = hr_df, scorer = rapidfuzz.fuzz.ratio, matching_function=parallel_rapidfuzz_match_with_score)
print(f"Elapsed time parallel&optimized rapidfuzz simple ratio: {(time.time() - start_time):.2f} seconds")
# Show results
# print(df_parallelopt_rapidfuzz_simple_ratio[['FirstName', 'MatchedFirstName', 'FirstNameScore', 'FirstNameIndex', 'LastName', 'MatchedLastName','LastNameScore','LastNameIndex','MatchedFullName','FullNameScore','FullNameIndex']].head(100))

# PARALLEL OPTIMIZED RAPIDFUZZ - PARITAL RATIO METHOD
start_time = time.time()
df_parallelopt_rapidfuzz_partial_ratio = best_match_with_score(param_df = hr_df, scorer = rapidfuzz.fuzz.partial_ratio, matching_function=parallel_rapidfuzz_match_with_score)
print(f"Elapsed time parallel&optimized rapidfuzz partial ratio: {(time.time() - start_time):.2f} seconds")
# Show results
# df_parallelopt_rapidfuzz_partial_ratio[['FirstName', 'MatchedFirstName', 'FirstNameScore', 'FirstNameIndex', 'LastName', 'MatchedLastName','LastNameScore','LastNameIndex','MatchedFullName','FullNameScore','FullNameIndex']].head(100)

# PARALLEL OPTIMIZED RAPIDFUZZ - TOKEN SORT RATIO METHOD
start_time = time.time()
df_parallelopt_rapidfuzz_token_sort_ratio = best_match_with_score(param_df = hr_df, scorer = rapidfuzz.fuzz.token_sort_ratio, matching_function=parallel_rapidfuzz_match_with_score)
print(f"Elapsed time parallel&optimized rapidfuzz token sort ratio: {(time.time() - start_time):.2f} seconds")
# Show results
# df_parallelopt_rapidfuzz_token_sort_ratio[['FirstName', 'MatchedFirstName', 'FirstNameScore', 'FirstNameIndex', 'LastName', 'MatchedLastName','LastNameScore','LastNameIndex','MatchedFullName','FullNameScore','FullNameIndex']].head(100)

# PARALLEL OPTIMIZED RAPIDFUZZ - TOKEN SET RATIO METHOD
start_time = time.time()
df_parallelopt_rapidfuzz_token_set_ratio = best_match_with_score(param_df = hr_df, scorer = rapidfuzz.fuzz.token_set_ratio, matching_function=parallel_rapidfuzz_match_with_score)
print(f"Elapsed time parallel&optimized rapidfuzz token set ratio: {(time.time() - start_time):.2f} seconds")
# print(f"df_parallelopt_rapidfuzz_token_set_ratio columns:{df_parallelopt_rapidfuzz_token_set_ratio.columns}")
# print(df_parallelopt_rapidfuzz_token_set_ratio[['FirstName', 'MatchedFirstName', 'FirstNameScore', 'FirstNameIndex', 'LastName', 'MatchedLastName','LastNameScore','LastNameIndex','MatchedFullName','FullNameScore','FullNameIndex']].head(100))
result_df = df_parallelopt_rapidfuzz_token_set_ratio.merge(client_df, how='left', left_on='MatchedFullName', right_on='FullName')
print(f"result_df columns:{result_df.columns}")
print(result_df.head(10))

# Saving merge/join results to result_df
result_df.to_csv(r"C:\Users\dalia\Desktop\Radu\DeutscheBank Fuzzy Matching\output_dataframes\merge_result_df.csv")