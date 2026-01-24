import sys
sys.path.insert(0, "/home/s232534/scripts")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
from sklearn.preprocessing import StandardScaler

from map_matching import routes

def run_pelt(df: pd.DataFrame, metric: str = 'SPEED_MS', visualise: bool = True, pelt_tuning: tuple[str, str, str] = (30, 10, 5)) -> list:
    if metric == 'SPEED_MS' or metric == "SPEED":
        signal = df[metric].values
    elif metric == 'multivariate': 
        features = ["SPEED_MS", "ACC_MS2", "BEARING_RATE_DEG_WGS84"]
        
        X = df[features].fillna(0).to_numpy()
        scaler = StandardScaler() # <- Scale it to avoid issue of features overpowering others by scale.
        X_std = scaler.fit_transform(X)

        w = np.array([1.0, 0.75, 0.5])  # weights currently hardcoded, can add a pass through var if needed.
        X_w = X_std * w
        signal = X_w
    elif metric == 'sb':
        features = ["SPEED_MS", "BEARING_RATE_DEG_WGS84"]
        
        X = df[features].fillna(0).to_numpy()
        scaler = StandardScaler() # <- Scale it to avoid issue of features overpowering others by scale.
        X_std = scaler.fit_transform(X)

        w = np.array([1.0, 1.0])  # weights just testing
        X_w = X_std * w
        signal = X_w
    else:
        signal = df[metric].fillna(0).values
    
    model = "rbf" #l1, l2, rbf are the types to use -> also currently hardcoded found the best segmentation results using rbf though.
    
    min_size, jump, pen = pelt_tuning
    
    # Sometimes the trip length may be shorter than the preset min_size and jump values
    # So we check if it is and if so we re-define those values to fit.
    # So if len(df) == 10: 10//2 = 5, max(2,5) == 5, min(30, 5) == 5. 
    min_size = int(min(min_size, max(2, len(df) // 2)))
    jump = int(min(jump, max(1, len(df) // 5)))
    
    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(signal)
    breakpoints = algo.predict(pen=pen) # Might need to make this adaptive depending on the tansport mode (the pen val)
    
    # Simple visualisation...
    if visualise:
        fig, ax_arr = rpt.display(signal, breakpoints, figsize=(10, 6))
        plt.show()
        
    return breakpoints

def run_pelt_signal(df: pd.DataFrame, signal_arr: np.array, visualise: bool = True, pelt_tuning: tuple[str, str, str] = (30, 10, 5)) -> list:
    if signal_arr is None:
        raise ValueError("signal_arr is None")
    
    signal_arr = np.asarray(signal_arr)
    
    if len(signal_arr) != len(df):
        raise ValueError(f"signal_arr length ({len(signal_arr)}) must match df length ({len(df)})")
    
    df_s = df.copy()
    df_s['SIGNAL'] = signal_arr
    features = ["SPEED_MS", "SIGNAL"]
    
    X = df_s[features].to_numpy()
    scaler = StandardScaler() # <- Scale it to avoid issue of features overpowering others by scale.
    X_std = scaler.fit_transform(X)

    w = np.array([1.0, 1.0])  # weights currently hardcoded, can add a pass through var if needed.
    X_w = X_std * w
    signal = X_w
    
    model = "rbf" #l1, l2, rbf are the types to use -> also currently hardcoded found the best segmentation results using rbf though.
    
    min_size, jump, pen = pelt_tuning
    
    # Sometimes the trip length may be shorter than the preset min_size and jump values
    # So we check if it is and if so we re-define those values to fit.
    # So if len(df) == 10: 10//2 = 5, max(2,5) == 5, min(30, 5) == 5. 
    min_size = int(min(min_size, max(2, len(df) // 2)))
    jump = int(min(jump, max(1, len(df) // 5)))
        
    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(signal)
    breakpoints = algo.predict(pen=pen) # Might need to make this adaptive depending on the tansport mode (the pen val)
    
    # Simple visualisation...
    if visualise:
        fig, ax_arr = rpt.display(signal, breakpoints, figsize=(10, 6))
        return breakpoints, fig, ax_arr
        #plt.show()
        
    return breakpoints

def run_pelt_signal_all_modes(df: pd.DataFrame, signal_arrs: tuple[np.array], main_mode: str, visualise: bool = True, pelt_tuning: tuple[str, str, str] = (30, 10, 5)) -> list:
    """
    Run multivariate PELT change-point detection using speed/bearing plus per-mode signal channels.

    Built from the trip dataframe and four external 
    mode-specific signal arrays (bus/train/subway/tram). Features are standardised and then
    re-weighted so the `main_mode` signal channel is more emphasised.

    Parameters
    ----------
    df : pd.DataFrame
        Trip dataframe. Must contain at least the columns ``SPEED_MS`` and
        ``BEARING_RATE_DEG_WGS84``.
    signal_arrs : tuple[np.array]
        Tuple of four arrays aligned to `df` rows, IN THE FOLLOWING SPECIFIC ORDER:
        ``(signal_bus, signal_train, signal_subway, signal_tram)``.
        Each array must be non-None and have length ``len(df)``.
    main_mode : str
        Main transport mode used to set adaptive feature weights. Expected values are
        ``'BUS'``, ``'TRAIN'``, ``'SUBWAY'``, or ``'TRAMWAY'``.
    visualise : bool, optional
        If True, returns the matplotlib figure/axes produced by ``ruptures.display``.
        If False, returns only the breakpoints.
    pelt_tuning : tuple[str, str, str], optional
        Tuple of ``(min_size, jump, pen)`` passed to PELT. Values are cast to ``int`` and
        ``min_size``/``jump`` are adapted for short trips.

    Returns
    -------
    list
        If ``visualise=False``: list of breakpoint indices.
        If ``visualise=True``: ``(breakpoints, fig, ax_arr)``.

    Raises
    ------
    ValueError
        If any signal array is None or if any signal array length does not match ``len(df)``.

    Notes
    -----
    - The feature set is: ``SPEED_MS``, ``BEARING_RATE_DEG_WGS84`` and four signal channels.
    - Standard scaling is applied before weighting to reduce scale dominance.
    """
    for arr in signal_arrs:
        if arr is None:
            raise ValueError("A signal_arr is None")
        if len(arr) != len(df):
            raise ValueError(f"A signal_arr length ({len(arr)}) must match df length ({len(df)})")
        
    signal_arr_bus, signal_arr_train, signal_arr_subway, signal_arr_tram = signal_arrs # explode
    
    signal_arr_bus = np.asarray(signal_arr_bus)
    signal_arr_train = np.asarray(signal_arr_train)
    signal_arr_subway = np.asarray(signal_arr_subway)
    signal_arr_tram = np.asarray(signal_arr_tram)
    
    df_s = df.copy()
    df_s['SIGNAL_BUS'] = signal_arr_bus
    df_s['SIGNAL_TRAIN'] = signal_arr_train
    df_s['SIGNAL_SUBWAY'] = signal_arr_subway
    df_s['SIGNAL_TRAM'] = signal_arr_tram

    # Adaptive scaling we will prioritise the heighest signal weight on the main_mode, as this meta-data assumption leads us to believe that the trip will spend the most time here!
    if main_mode == 'BUS':
        ws = np.array([1.0, 1.0, 1.0, 0.75, 0.75, 0.75])
    elif main_mode in ("TRAIN", "TRAIN_EXPRESS"):
        ws = np.array([1.0, 1.0, 0.75, 1.0, 0.75, 0.75])
    elif main_mode == 'SUBWAY':
        ws = np.array([1.0, 1.0, 0.75, 0.75, 1.0, 0.75])
    elif main_mode == 'TRAMWAY':
        ws = np.array([1.0, 1.0, 0.75, 0.75, 0.75, 1.0])
    else:
        # It shouldn't reach here if upstream we have a check for what modes can use this function
        # In another iteration we do reach here... 
        #ws = np.ones(6) 
        ws = np.array([1.0, 1.0, 0.75, 0.75, 0.75, 0.75])
    # we split out the continuous and binary signals to two seperate lists... 
    # This is because we do not what to normalise a binary signal.
    # We only normalise speed and bearing rate and then concat back together and apply the weights.
    
    cont = ["SPEED_MS", "BEARING_RATE_DEG_WGS84"]
    bins = ["SIGNAL_BUS", "SIGNAL_TRAIN", "SIGNAL_SUBWAY", "SIGNAL_TRAM"]

    Xc = df_s[cont].fillna(0).to_numpy()
    Xb = df_s[bins]

    scaler = StandardScaler()
    Xc_std = scaler.fit_transform(Xc)
    
    signal = np.hstack([Xc_std, Xb])
    signal = signal * ws
    
    model = "rbf" #l1, l2, rbf are the types to use -> also currently hardcoded found the best segmentation results using rbf though.
    
    min_size, jump, pen = pelt_tuning
    
    # Sometimes the trip length may be shorter than the preset min_size and jump values
    # So we check if it is and if so we re-define those values to fit.
    # So if len(df) == 10: 10//2 = 5, max(2,5) == 5, min(30, 5) == 5. 
    min_size = int(min(min_size, max(2, len(df) // 2)))
    jump = int(min(jump, max(1, len(df) // 5)))
        
    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(signal)
    breakpoints = algo.predict(pen=pen) # Might need to make this adaptive depending on the tansport mode (the pen val)
    
    # Simple visualisation...
    if visualise:
        fig, ax_arr = rpt.display(signal, breakpoints, figsize=(10, 6))
        return breakpoints, fig, ax_arr
        #plt.show()
        
    return breakpoints

def segment_trip_dataframe(df: pd.DataFrame, pelt_segment_points: list, visualise: bool = True) -> pd.DataFrame:
    """
    Assign segment labels to a trip DataFrame from PELT breakpoints and optionally plot speed by segment.

    Parameters
    ----------
    df : pd.DataFrame
        Input trip data with at least a 'SPEED_MS' column. The col is only for the visualisation.
    pelt_segment_points : list
        Ordered list of breakpoint indices returned by PELT.
    visualise : bool, optional
        If True, plot 'SPEED_MS' for each segment, by default True.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with an added 'CHANGE_POINT' integer segment label column.
    """
    
    df_seg = df.copy()
    df_seg['CHANGE_POINT'] = 0

    # init array 
    arr = np.zeros(len(df_seg)) 

    previous_point = 0
    for i, point in enumerate(pelt_segment_points):
        arr[previous_point:point] = i
        previous_point = point

    # Set the col with our array values:    
    df_seg['CHANGE_POINT'] = arr.astype('int')
    
    # Simple vis
    if visualise:
        for seg_id, seg_df in df_seg.groupby('CHANGE_POINT'):
            df_seg['SPEED_MS'].plot()
            
    return df_seg