import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

# Paths
GPS_DIR = Path("/home/lauale/netmob2025/data/gps_dataset")
TRIPS_PATH = Path("/home/lauale/netmob2025/data/trips_dataset.csv")
OUT_DIR = Path("/home/s232534/data/split")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load trips ---
df_trips = pd.read_csv(TRIPS_PATH)

# Fix up the DF dtypes:
#df_trips['Date_O'] = pd.to_datetime(df_trips['Date_O'], errors='coerce')
#df_trips['Date_D'] = pd.to_datetime(df_trips['Date_D'], errors='coerce')
#df_trips['Date_EMG'] = pd.to_datetime(df_trips['Date_EMG'], errors='coerce')

# Need to check if dates are alligned otherwise it won't match correctly for example there is an issue
# with ID 49_1414 -> where the date is 2022-12-05 but on the Date_O and Date_D it is 2022-05-12
# meaning that it thinks its the 12th of May instead of the 5th of December!
# the only problem it is hard to tell when we should do the fallback... like which date is then correct?
# Do I take the EMG date? -> for now this will be my solution.
# Maybe you should discard this data for robustness because we cannot infer which date is correct?
# if (df_trips["Date_EMG"] == df_trips['Date_O'] and df_trips["Date_EMG"] == df_trips['Date_D']):
#     df_trips['start_time'] = pd.to_datetime(df_trips['Date_O'] + " " + df_trips['Time_O'], errors='coerce')
#     df_trips['end_time'] = pd.to_datetime(df_trips['Date_D'] + " " + df_trips['Time_D'], errors='coerce')
#     df_trips = df_trips.dropna(subset=['start_time', 'end_time'])
# else:
#     df_trips['start_time'] = pd.to_datetime(df_trips['Date_EMG'] + " " + df_trips['Time_O'], errors='coerce')
#     df_trips['end_time'] = pd.to_datetime(df_trips['Date_EMG'] + " " + df_trips['Time_D'], errors='coerce')
#     df_trips = df_trips.dropna(subset=['start_time', 'end_time'])

# Boolean mask: True where EMG == O == D
# mask = (df_trips["Date_EMG"] == df_trips["Date_O"]) & (df_trips["Date_EMG"] == df_trips["Date_D"])

# # Case 1: aligned (use Date_O and Date_D)
# df_trips.loc[mask, "start_time"] = pd.to_datetime(df_trips.loc[mask, "Date_O"].astype(str) + " " + df_trips.loc[mask, "Time_O"], errors="coerce")
# df_trips.loc[mask, "end_time"] = pd.to_datetime(df_trips.loc[mask, "Date_D"].astype(str) + " " + df_trips.loc[mask, "Time_D"], errors="coerce")

# # Case 2: misaligned (fallback to Date_EMG)
# df_trips.loc[~mask, "start_time"] = pd.to_datetime(df_trips.loc[~mask, "Date_EMG"].astype(str) + " " + df_trips.loc[~mask, "Time_O"], errors="coerce")
# df_trips.loc[~mask, "end_time"] = pd.to_datetime(df_trips.loc[~mask, "Date_EMG"].astype(str) + " " + df_trips.loc[~mask, "Time_D"], errors="coerce")

# Try just using Date_EMG always!
df_trips['start_time'] = pd.to_datetime(df_trips['Date_EMG'] + " " + df_trips['Time_O'], errors='coerce')
df_trips['end_time'] = pd.to_datetime(df_trips['Date_EMG'] + " " + df_trips['Time_D'], errors='coerce')
df_trips = df_trips.dropna(subset=['start_time', 'end_time'])

# Fix trips crossing midnight -> bit of a bandaid fix maybe?
mask = df_trips['end_time'] < df_trips['start_time']
df_trips.loc[mask, 'end_time'] += pd.Timedelta(days=1)

def process_one(csv_path: Path) -> None:
    # Filter trips only for current ID
    cur_id = csv_path.stem
    df_trips_inside = df_trips[df_trips['ID'] == cur_id]

    # If no trips, still produce an output with no KEY and record all missing? Keep simple: write GPS as-is and a missing file.
    # --- Load GPS ---
    
    # Had problems with some GPS data missing/empty so needed to add checks!
    try:
        df_gps = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"Skipping {csv_path.name}: empty file")
        return

    if df_gps.empty:
        print(f"Skipping {csv_path.name}: no rows")
        return

    if 'LOCAL DATETIME' not in df_gps.columns:
        print(f"Skipping {csv_path.name}: missing LOCAL DATETIME")
        return

    # Ensure a clean index and pre-create KEY to avoid assignment edge cases
    df_gps = df_gps.reset_index(drop=True)
    df_gps['timestamp'] = pd.to_datetime(df_gps['LOCAL DATETIME'], errors='coerce')
    # Ensure KEY exists and treat empty/blank as missing (NaN)
    if 'KEY' in df_gps.columns:
        df_gps['KEY'] = (df_gps['KEY'].astype('string').str.strip().replace({'': pd.NA, 'nan': pd.NA, 'NaN': pd.NA}))
    else:
        df_gps['KEY'] = pd.Series(pd.NA, index=df_gps.index, dtype='string')

    # Assign KEY by checking if timestamp is within each trip interval
    for _, trip in df_trips_inside.iterrows():
        mask = (df_gps['timestamp'] >= trip['start_time']) & (df_gps['timestamp'] <= trip['end_time'])
        df_gps.loc[mask, 'KEY'] = trip['KEY']

    # CORRECT DATATYPES TO REDUCE CSV SIZE!
    # WAS WRONG ABOUT THIS... CSV OF COURSE WON'T PRESERVE A DF DTYPE...
    # df_gps["UTC DATETIME"] = pd.to_datetime(df_gps['UTC DATETIME'], errors='coerce')
    # df_gps["LOCAL DATETIME"] = pd.to_datetime(df_gps['LOCAL DATETIME'], errors='coerce')
    # df_gps["timestamp"] = pd.to_datetime(df_gps['timestamp'], errors='coerce')
    # df_gps["KEY"] = df_gps["KEY"].astype("string")
    # df_gps["VALID"] = df_gps["VALID"].astype("string")

    # Save per-file GPS with KEYs
    out_path = OUT_DIR / f"{csv_path.stem}_s.csv"
    # Write 'NaN' explicitly for missing KEY values only; keep other NaNs as empty
    #df_out = df_gps.copy()
    df_gps['KEY'] = df_gps['KEY'].fillna('NaN')
    df_gps.to_csv(out_path, index=False)
    
    trip_keys = set(df_trips.loc[df_trips["ID"] == csv_path.stem, "KEY"].unique())
    # Get the set of trip keys that actually appear in GPS after merging
    gps_keys = set(df_gps["KEY"].dropna().unique())
    missing = trip_keys - gps_keys
    
    if missing:
        miss_path = OUT_DIR / f"missing_trips__{csv_path.stem}.csv"
        # Include the associated ID for each missing KEY, along with start and end times
        df_missing = (
            df_trips_inside[df_trips_inside["KEY"].isin(missing)]
            .drop_duplicates(subset=["KEY"])  # ensure one row per KEY
            .dropna(subset=["start_time", "end_time"]) # drop the missing journey's that have no time because they are already accounted for as missing in the trips_dataset.
            .loc[:, ["ID", "KEY", "start_time", "end_time"]]
            .rename(columns={"KEY": "missing_KEY"})
            .sort_values("missing_KEY")
        )
        df_missing.to_csv(miss_path, index=False)


def main(limit: int | None = None, workers: int = 1, id_prefix: str = "") -> None:
    # Collect files
    #csvs = sorted(GPS_DIR.glob(f"{id_prefix}*.csv"))
    csvs = sorted(GPS_DIR.glob(f"*.csv"))
    #print(csvs[300]) #DEBUG
    if limit is not None:
        csvs = csvs[:limit]

    if not csvs:
        print("No CSVs found to process.")
        return

    os.environ.setdefault("OMP_NUM_THREADS", "1")

    if workers == 1:
        #for p in csvs:
        for p in tqdm(csvs, desc="Processing", unit="File"):
            process_one(p)
            
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            #list(ex.map(process_one, csvs))
            list(tqdm(ex.map(process_one, csvs), total=len(csvs), desc="Processing", unit="file"))

    # Merge missing trip reports
    miss_parts = list(OUT_DIR.glob("missing_trips__*.csv")) # TODO: Maybe a bug here wiht double underscores -> hasn't seemed to have an affect though?
    if miss_parts:
        (pd.concat((pd.read_csv(p) for p in miss_parts), ignore_index=True)
           .drop_duplicates()
           .to_csv(OUT_DIR / "missing_trips.csv", index=False))
        for p in miss_parts:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    main(limit=None, workers=os.cpu_count() or 4, id_prefix="")
