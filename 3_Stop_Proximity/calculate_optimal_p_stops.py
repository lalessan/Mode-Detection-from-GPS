import numpy as np
import pandas as pd
from itertools import combinations

# NOTE: This is the same template as found in percentile_calculate.py
# TODO: For future merge the functionality of the two into one script/library as there is a lot of redundancy.

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

def calculate_percentile_distributions(df_pm: pd.DataFrame, bins: int = 50, mode: str = 'mean') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    # Single mode journeys only!
    df_pm = df_pm[df_pm['MULTI_MODAL'] == False]

    # Filter the dataframe primarily for the missing values which occurs when we have many stacked coords and very few GPS pings. (Pattern: usually happens for WALKING)
    # Exclude rows where BC_P50, BR_WGS84_P50, or BR_P50 are NaN this will mean they are all NaN and skew the distribution.
    df_pm = df_pm.dropna(subset=['STOPS_CLOSEST_TRAM_P50']) # In theory if one is NaN all would be so we just check for one val and drop col if Na
    
    modes = df_pm['MAIN_MODE'].unique() # <- Get all transport mode labels.
    columns_percentile_map = df_pm.columns[df_pm.columns.str.contains(r'_P\d+$')]
    
    # Creates a dictionary of the edge values to use to generate the histograms inside a col
    dict_prob_denisty = {}
    js_values = []
    #df_JS_per_percentile_mode = pd.DataFrame(0.0, index=modes, columns=modes)
    for col in columns_percentile_map:
        edges = _make_pooled_edges(df_pm, col, bins=bins, trim=(0.01, 0.99))

        dict_prob_denisty[col] = {} # init dict
        for m in modes:
            vals = df_pm.loc[df_pm['MAIN_MODE'] == m, col].values
            dict_prob_denisty[col][m] = _histogram_probability_normalised(vals, edges)

        for a, b in combinations(modes, 2):
            js = _jensen_shannon_divergence(dict_prob_denisty[col][a], dict_prob_denisty[col][b])
            js_values.append({'col': col, 'mode_a': a, 'mode_b': b, 'js': js})
    
    df_all_js = pd.DataFrame(js_values)        

    df_js_aggregate = df_all_js.groupby(['col'])['js'].agg(['mean', 'std', 'median', 'min', 'max'])
    df_js_aggregate = df_js_aggregate.reset_index()
    
    df_js_aggregate['feature'] = df_js_aggregate['col'].str.split('_P').str[0]
    
    best_percentiles = df_js_aggregate.loc[df_js_aggregate.groupby('feature')[mode].idxmax()]

    print(best_percentiles[['feature', 'col', 'mean', 'std']])
    
    return best_percentiles, df_js_aggregate, df_all_js
    #DEBUG
    # plt.plot(df_js_aggregate[df_js_aggregate['feature']=='SPEED']['col'],df_js_aggregate[df_js_aggregate['feature']=='SPEED']['mean'], 'o-')
    # plt.show()
