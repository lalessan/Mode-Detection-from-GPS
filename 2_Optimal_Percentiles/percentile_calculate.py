import sys
sys.path.insert(0, "/home/s232534/scripts")

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import combinations
from exploration import functions # This is a custom script and may require the above system path to point to where it is contained

#PERCENTILE_MAP_PATH = Path("C:/Users/Mikus/Documents/DTU/Thesis/BackupServer/percentiles_231025.csv")

THRESHOLDS: dict = { 
        "WALKING": 6.5, # under 25 km/h
        "BIKE": 10, # 36 km/h
        "CAR": 39,
        "BUS": 14,  # ~50 km/h      
        "TRAMWAY": 19.5,     # ~70 km/h
        "SUBWAY": 11.1, # 40 km/h
        "TRAIN": 61,  # ~ 220 km/h     
        "ELECT_SCOOTER": 10,  
        "ON_DEMAND": 39,      
        "TWO_WHEELER": 39,
        "ELECT_BIKE": 14,      # ~50 km/h
    }

MODE_REMAP = {
        'PRIV_CAR_DRIVER': 'CAR',
        'PRIV_CAR_PASSENGER': 'CAR',
        'LIGHT_COMM_VEHICLE': 'CAR',
        'TAXI': 'CAR',
        'TRAIN_EXPRESS': 'TRAIN'
    }

MODE_DROP = ['OTHER', 'PLANE', 'ON_DEMAND', 'ELECT_BIKE', 'ELECT_SCOOTER', 'TWO_WHEELER']

def  _kullback_leibler_divergence(P, Q):
    epsilon = 1e-12
    # Can also consider using np.eps in this config -> np.finfo(float).eps | but from what I read it is a less of a "stable" number.
    P = np.clip(P, epsilon, None)
    Q = np.clip(Q, epsilon, None)
    
    return np.sum(P * (np.log(P) - np.log(Q)))

def _jensen_shannon_divergence(P, Q):
    M = 0.5 * (P + Q) # This is the average distribution of the two.
    
    return (0.5 * _kullback_leibler_divergence(P, M)) + (0.5 * _kullback_leibler_divergence(Q, M))

def _make_pooled_edges(df, col, bins=50, trim=(0.01,0.99)):
    """This is so that we get the same binning for all modes in the col (percentile col of certain motion feature)
    Avoids the issue where if we create edges from per mode we cannot us JS to compare the divergence, so we have to ensure standardisation across all modes.

    Args:
        df (pd.DataFrame): _description_
        col (Percentile motion feature col): _description_
        bins (int, optional): _description_. Defaults to 50.
        trim (tuple, optional): _description_. Defaults to (0.01,0.99).

    Returns:
        list[edges: float]: 
    """
    x = df[col].dropna().to_numpy()
    if trim:
        # Allows us to avoid large GPS glitches/outliers
        lo, hi = np.quantile(x, trim)
    else:
        lo, hi = np.min(x), np.max(x)
    if hi <= lo:
        return np.nan
    return np.linspace(lo, hi, bins + 1)

def _histogram_probability_normalised(values, edges):
    counts, _ = np.histogram(values, bins=edges, density=False)
    total = counts.sum()

    return counts / total

def calculate_percentile_distributions(df_pm: pd.DataFrame, remap: bool = True, mode: str = 'mean') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculates Jensen-Shannon divergence between transport modes across motion feature percentiles
    to identify the largest density seperation between modes of travel for a certain percentile.
    
    This function:
    1. Loads precomputed percentile motion features (percentile map)
    2. Filters to single-mode journeys and removes invalid data (optional)
    3. Computes probability distributions for each mode at each percentile level
    4. Calculates Jensen-Shannon divergence for all mode pairs at each percentile
    5. Aggregates JS divergence statistics to identify optimal percentiles per feature
    
    The analysis covers 75 percentile columns across 7 motion features:
    - SPEED, ACC, JERK, BR, BR_WGS84, BC, BC_WGS84
    - Each at percentiles P50, P55, P60, P65, P70, P75, P80, P85, P90, P95

    Args:
        remap (bool, optional): Whether to remap and drop certain transport modes. 
                                Consolidates similar modes (e.g., PRIV_CAR_DRIVER -> CAR)
                                and removes rarely used modes. Defaults to True.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            - best_percentiles: DataFrame with the optimal percentile per feature by the mean 
                               (highest mean JS divergence)
            - df_js_aggregate: DataFrame with aggregated statistics (mean, std, median, 
                              min, max) of JS divergence for each percentile column (used to calculate the best_percentile)
            - df_all_js: DataFrame with individual JS divergence values for all 
                        mode pairs at each percentile (essentially the raw data after calculations are made)
    """
    
    # Load our precomputed percentile map
    #df_pm = pd.read_csv(PERCENTILE_MAP_PATH)
    
    # Single mode journeys only!
    df_pm = df_pm[df_pm['MULTI_MODAL'] == False]

    # Filter the dataframe primarily for the missing values which occurs when we have many stacked coords and very few GPS pings. (Pattern: usually happens for WALKING)
    # Exclude rows where BC_P50, BR_WGS84_P50, or BR_P50 are NaN this will mean they are all NaN and skew the distribution.
    df_pm = df_pm.dropna(subset=['BC_P50', 'BR_WGS84_P50', 'BR_P50', 'BC_WGS84_P50'])
    
    #TODO: Can't actually do this because the percentile map does not contain averages... either add that to the percentile map or filter by linking to the motion features dataset.
    
    # Remap modes and drop modes which we don't need.
    if (remap):
        df_pm = functions.remap_and_drop_modes_df(df_pm, MODE_REMAP, MODE_DROP)
    
    modes = df_pm['MAIN_MODE'].unique() # <- Get all transport mode labels.
    columns_percentile_map = ['SPEED_P50', 'SPEED_P55', 'SPEED_P60', 'SPEED_P65', 'SPEED_P70', 'SPEED_P75', 'SPEED_P80', 'SPEED_P85', 'SPEED_P90', 'SPEED_P95', 'ACC_P50', 'ACC_P55', 'ACC_P60', 'ACC_P65', 'ACC_P70', 'ACC_P75', 'ACC_P80', 'ACC_P85', 'ACC_P90', 'ACC_P95', 'JERK_P50', 'JERK_P55', 'JERK_P60', 'JERK_P65', 'JERK_P70', 'JERK_P75', 'JERK_P80', 'JERK_P85', 'JERK_P90', 'JERK_P95', 'BR_P50', 'BR_P55', 'BR_P60', 'BR_P65', 'BR_P70', 'BR_P75', 'BR_P80', 'BR_P85', 'BR_P90', 'BR_P95', 'BR_WGS84_P50', 'BR_WGS84_P55', 'BR_WGS84_P60', 'BR_WGS84_P65', 'BR_WGS84_P70', 'BR_WGS84_P75', 'BR_WGS84_P80', 'BR_WGS84_P85', 'BR_WGS84_P90', 'BR_WGS84_P95', 'BC_P50', 'BC_P55', 'BC_P60', 'BC_P65', 'BC_P70', 'BC_P75', 'BC_P80', 'BC_P85', 'BC_P90', 'BC_P95', 'BC_WGS84_P50', 'BC_WGS84_P55', 'BC_WGS84_P60', 'BC_WGS84_P65', 'BC_WGS84_P70', 'BC_WGS84_P75', 'BC_WGS84_P80', 'BC_WGS84_P85', 'BC_WGS84_P90', 'BC_WGS84_P95']
    
    # Creates a dictionary of the edge values to use to generate the histograms inside a col
    dict_prob_denisty = {}
    js_values = []
    #df_JS_per_percentile_mode = pd.DataFrame(0.0, index=modes, columns=modes)
    for col in columns_percentile_map:
        edges = _make_pooled_edges(df_pm, col, bins=50, trim=(0.01, 0.99))
        #col_edges[col] = edges

        dict_prob_denisty[col] = {} # init dict
        for m in modes:
            vals = df_pm.loc[df_pm['MAIN_MODE'] == m, col].values
            dict_prob_denisty[col][m] = _histogram_probability_normalised(vals, edges)
            # centers = 0.5 * (edges[1:] + edges[:-1])
            # plt.bar(centers, _histogram_probability_normalised(vals, edges), width=np.diff(edges), label='test', align='center')

        for a, b in combinations(modes, 2):
            js = _jensen_shannon_divergence(dict_prob_denisty[col][a], dict_prob_denisty[col][b])
            # df_JS_per_percentile_mode.loc[a, b] = js
            # df_JS_per_percentile_mode.loc[b, a] = js
            js_values.append({'col': col, 'mode_a': a, 'mode_b': b, 'js': js})
    
    df_all_js = pd.DataFrame(js_values)        
    # v1 = _jensen_shannon_divergence(dict_prob_denisty['SPEED_P75']['CAR'], dict_prob_denisty['SPEED_P75']['BIKE'])
    # print(v1)
    
    print(df_all_js.head(50))
    
    #df_js_aggregate = df_all_js.groupby(['col'])['js'].mean()
    df_js_aggregate = df_all_js.groupby(['col'])['js'].agg(['mean', 'std', 'median', 'min', 'max'])
    df_js_aggregate = df_js_aggregate.reset_index()
    
    df_js_aggregate['feature'] = df_js_aggregate['col'].str.split('_P').str[0]
    print(df_js_aggregate.head(50))
    
    best_percentiles = df_js_aggregate.loc[df_js_aggregate.groupby('feature')[mode].idxmax()]

    print(best_percentiles[['feature', 'col', 'mean', 'std']])
    
    return best_percentiles, df_js_aggregate, df_all_js
    #DEBUG
    # plt.plot(df_js_aggregate[df_js_aggregate['feature']=='SPEED']['col'],df_js_aggregate[df_js_aggregate['feature']=='SPEED']['mean'], 'o-')
    # plt.show()

def calculate_percentile_distributions_segmented(df_pm: pd.DataFrame, remap: bool = True, mode: str = 'mean') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    NOTE: This function is effectively a clone of the above with some minor col changes for segmented pipeline.
    
    Calculates Jensen-Shannon divergence between transport modes across motion feature percentiles
    to identify the largest density seperation between modes of travel for a certain percentile.
    
    This function:
    1. Loads precomputed percentile motion features (percentile map)
    2. Filters to single-mode journeys and removes invalid data (optional)
    3. Computes probability distributions for each mode at each percentile level
    4. Calculates Jensen-Shannon divergence for all mode pairs at each percentile
    5. Aggregates JS divergence statistics to identify optimal percentiles per feature
    
    The analysis covers 75 percentile columns across 7 motion features:
    - SPEED, ACC, JERK, BR, BR_WGS84, BC, BC_WGS84
    - Each at percentiles P50, P55, P60, P65, P70, P75, P80, P85, P90, P95

    Args:
        remap (bool, optional): Whether to remap and drop certain transport modes. 
                                Consolidates similar modes (e.g., PRIV_CAR_DRIVER -> CAR)
                                and removes rarely used modes. Defaults to True.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            - best_percentiles: DataFrame with the optimal percentile per feature by the mean 
                               (highest mean JS divergence)
            - df_js_aggregate: DataFrame with aggregated statistics (mean, std, median, 
                              min, max) of JS divergence for each percentile column (used to calculate the best_percentile)
            - df_all_js: DataFrame with individual JS divergence values for all 
                        mode pairs at each percentile (essentially the raw data after calculations are made)
    """
    # Filter the dataframe primarily for the missing values which occurs when we have many stacked coords and very few GPS pings. (Pattern: usually happens for WALKING)
    # Exclude rows where BC_P50, BR_WGS84_P50, or BR_P50 are NaN this will mean they are all NaN and skew the distribution.
    df_pm = df_pm.dropna(subset=['BR_WGS84_P50', 'BC_WGS84_P50'])
    
    # Remap modes and drop modes which we don't need.
    if (remap):
        df_pm = functions.remap_and_drop_modes_df(df_pm, MODE_REMAP, MODE_DROP)
    
    modes = df_pm['MAIN_MODE'].unique() # <- Get all transport mode labels.
    columns_percentile_map = ['SPEED_P50', 'SPEED_P55', 'SPEED_P60', 'SPEED_P65', 'SPEED_P70', 'SPEED_P75', 'SPEED_P80', 'SPEED_P85', 'SPEED_P90', 'SPEED_P95', 
                              'ACC_P50', 'ACC_P55', 'ACC_P60', 'ACC_P65', 'ACC_P70', 'ACC_P75', 'ACC_P80', 'ACC_P85', 'ACC_P90', 'ACC_P95', 
                              'JERK_P50', 'JERK_P55', 'JERK_P60', 'JERK_P65', 'JERK_P70', 'JERK_P75', 'JERK_P80', 'JERK_P85', 'JERK_P90', 'JERK_P95', 
                              'BR_WGS84_P50', 'BR_WGS84_P55', 'BR_WGS84_P60', 'BR_WGS84_P65', 'BR_WGS84_P70', 'BR_WGS84_P75', 'BR_WGS84_P80', 'BR_WGS84_P85', 'BR_WGS84_P90', 'BR_WGS84_P95', 
                              'BC_WGS84_P50', 'BC_WGS84_P55', 'BC_WGS84_P60', 'BC_WGS84_P65', 'BC_WGS84_P70', 'BC_WGS84_P75', 'BC_WGS84_P80', 'BC_WGS84_P85', 'BC_WGS84_P90', 'BC_WGS84_P95']
    
    # Creates a dictionary of the edge values to use to generate the histograms inside a col
    dict_prob_denisty = {}
    js_values = []

    for col in columns_percentile_map:
        edges = _make_pooled_edges(df_pm, col, bins=50, trim=(0.01, 0.99))

        dict_prob_denisty[col] = {} # init dict
        for m in modes:
            vals = df_pm.loc[df_pm['MAIN_MODE'] == m, col].values
            dict_prob_denisty[col][m] = _histogram_probability_normalised(vals, edges)

        for a, b in combinations(modes, 2):
            js = _jensen_shannon_divergence(dict_prob_denisty[col][a], dict_prob_denisty[col][b])
            js_values.append({'col': col, 'mode_a': a, 'mode_b': b, 'js': js})
    
    df_all_js = pd.DataFrame(js_values)        
    
    print(df_all_js.head(50))
    
    df_js_aggregate = df_all_js.groupby(['col'])['js'].agg(['mean', 'std', 'median', 'min', 'max'])
    df_js_aggregate = df_js_aggregate.reset_index()
    
    df_js_aggregate['feature'] = df_js_aggregate['col'].str.split('_P').str[0]
    print(df_js_aggregate.head(50))
    
    best_percentiles = df_js_aggregate.loc[df_js_aggregate.groupby('feature')[mode].idxmax()]

    print(best_percentiles[['feature', 'col', 'mean', 'std']])
    
    return best_percentiles, df_js_aggregate, df_all_js
 