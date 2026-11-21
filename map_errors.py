"""Plot misclassified trips on interactive maps.

For each major confusion pair, picks example trips and creates a folium
HTML map showing the GPS trace colored by speed.

Usage:
    python map_errors.py
    # Open the HTML files in a browser
"""
import numpy as np
import pandas as pd
import folium
from pathlib import Path

CHECKPOINT_DIR = Path("checkpoints/")
OUT_DIR = Path("error_maps/")
OUT_DIR.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────

print("Loading OOF predictions...")
oof = pd.read_parquet(CHECKPOINT_DIR / "fit_steps" / "05_oof_predictions.parquet")
true_col, pred_col = "true_label", "pred_label"

print("Loading GPS data...")
gps = pd.read_parquet(CHECKPOINT_DIR / "fit_steps" / "01_gps_motion.parquet")

# Load stats for feature context
stats = pd.read_parquet(CHECKPOINT_DIR / "fit_steps" / "02_stats.parquet")

# Merge predictions with stats
oof_stats = oof.merge(stats, on="trip_key", how="left")

# ── Define confusion pairs to visualize ───────────────────────────

PAIRS = [
    ("SUBWAY", "WALKING", 5),   # 5 examples each
    ("SUBWAY", "BUS", 3),
    ("CAR", "WALKING", 5),
    ("CAR", "BUS", 3),
    ("WALKING", "SUBWAY", 3),
    ("BUS", "CAR", 3),
]

# Also show correctly classified examples for comparison
CORRECT_EXAMPLES = ["SUBWAY", "BUS", "CAR", "WALKING"]


def make_trip_map(trip_gps, trip_key, true_mode, pred_mode, stats_row=None):
    """Create a folium map for a single trip."""
    lat_col, lon_col = "LATITUDE", "LONGITUDE"
    speed_col = "SPEED"

    lats = trip_gps[lat_col].values
    lons = trip_gps[lon_col].values
    speeds = trip_gps[speed_col].values if speed_col in trip_gps.columns else np.zeros(len(trip_gps))

    center_lat = np.median(lats)
    center_lon = np.median(lons)

    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=14,
        tiles="CartoDB positron",
    )

    # Color by speed: blue=slow, red=fast
    max_speed = max(speeds.max(), 1)
    coords = list(zip(lats, lons))

    # Draw line segments colored by speed
    for i in range(len(coords) - 1):
        speed = speeds[i]
        # Normalize: 0=blue, >5 m/s = red
        t = min(speed / 5.0, 1.0)
        r = int(255 * t)
        g = 0
        b = int(255 * (1 - t))
        color = f"#{r:02x}{g:02x}{b:02x}"

        folium.PolyLine(
            [coords[i], coords[i + 1]],
            color=color, weight=3, opacity=0.8,
        ).add_to(m)

    # Start/end markers
    folium.Marker(coords[0], popup="START", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(coords[-1], popup="END", icon=folium.Icon(color="red")).add_to(m)

    # Title
    info = f"True: {true_mode} | Pred: {pred_mode}"
    if stats_row is not None:
        speed_avg = stats_row.get("SPEED_AVG", 0)
        total_time = stats_row.get("TOTAL_TIME", 0)
        gap_max = stats_row.get("GAP_MAX", 0)
        dist = stats_row.get("DIST_TOTAL_WGS84", 0)
        subway_pct = stats_row.get("subway_match_pct", 0)
        bus_pct = stats_row.get("bus_match_pct", 0)
        info += (f"<br>Speed avg: {speed_avg:.1f} m/s, max gap: {gap_max:.0f}s"
                 f"<br>Duration: {total_time:.0f}s, Dist: {dist:.0f}m"
                 f"<br>subway_match: {subway_pct:.2f}, bus_match: {bus_pct:.2f}")

    title_html = f'''
    <div style="position:fixed; top:10px; left:60px; z-index:9999;
                background:white; padding:10px; border:2px solid black;
                font-size:14px; font-family:monospace;">
        <b>{trip_key}</b><br>{info}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Legend
    legend_html = '''
    <div style="position:fixed; bottom:30px; left:60px; z-index:9999;
                background:white; padding:8px; border:1px solid grey;
                font-size:12px;">
        <b>Speed</b><br>
        <span style="color:#0000ff;">■</span> Slow (0 m/s)&nbsp;
        <span style="color:#ff0000;">■</span> Fast (5+ m/s)
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ── Load route features for context ──────────────────────────────

routes_path = CHECKPOINT_DIR / "fit_steps" / "04_route_features.parquet"
routes = pd.read_parquet(routes_path) if routes_path.exists() else None
if routes is not None:
    oof_stats = oof_stats.merge(routes, on="trip_key", how="left", suffixes=("", "_route"))

stops_path = CHECKPOINT_DIR / "fit_steps" / "03_stop_features.parquet"
stops = pd.read_parquet(stops_path) if stops_path.exists() else None
if stops is not None:
    # Only keep key columns to avoid bloat
    stop_cols = ["trip_key", "MIN_DIST_METRO", "DURATION_METRO", "MIN_DIST_BUS", "DURATION_BUS"]
    stop_cols = [c for c in stop_cols if c in stops.columns]
    oof_stats = oof_stats.merge(stops[stop_cols], on="trip_key", how="left", suffixes=("", "_stop"))

# ── Generate maps ─────────────────────────────────────────────────

all_maps = []

# Confusion pairs
for true_mode, pred_mode, n_examples in PAIRS:
    mask = (oof_stats[true_col] == true_mode) & (oof_stats[pred_col] == pred_mode)
    confused = oof_stats[mask].head(n_examples)

    if len(confused) == 0:
        print(f"  No examples for {true_mode} → {pred_mode}")
        continue

    print(f"\n  {true_mode} → {pred_mode}: {len(confused)} examples")

    for _, row in confused.iterrows():
        trip_key = row["trip_key"]
        trip_gps = gps[gps["KEY"] == trip_key].sort_values("timestamp")
        if len(trip_gps) == 0:
            print(f"    {trip_key}: no GPS data found")
            continue

        m = make_trip_map(trip_gps, trip_key, true_mode, pred_mode, row)
        fname = f"{true_mode}_to_{pred_mode}_{trip_key.replace('/', '_')}.html"
        fpath = OUT_DIR / fname
        m.save(str(fpath))
        all_maps.append((true_mode, pred_mode, trip_key, fpath))
        print(f"    Saved: {fpath}")

# Correctly classified examples for comparison
for mode in CORRECT_EXAMPLES:
    mask = (oof_stats[true_col] == mode) & (oof_stats[pred_col] == mode)
    correct = oof_stats[mask].head(3)

    print(f"\n  Correct {mode}: {len(correct)} examples")

    for _, row in correct.iterrows():
        trip_key = row["trip_key"]
        trip_gps = gps[gps["KEY"] == trip_key].sort_values("timestamp")
        if len(trip_gps) == 0:
            continue

        m = make_trip_map(trip_gps, trip_key, mode, mode, row)
        fname = f"CORRECT_{mode}_{trip_key.replace('/', '_')}.html"
        fpath = OUT_DIR / fname
        m.save(str(fpath))
        all_maps.append((mode, mode, trip_key, fpath))
        print(f"    Saved: {fpath}")

# ── Create index page ─────────────────────────────────────────────

index_html = "<html><head><title>Error Analysis Maps</title></head><body>\n"
index_html += "<h1>Misclassified Trip Maps</h1>\n"
index_html += "<p>Blue = slow, Red = fast. Green marker = start, Red marker = end.</p>\n"

current_pair = None
for true_mode, pred_mode, trip_key, fpath in all_maps:
    pair = f"{true_mode} → {pred_mode}"
    if pair != current_pair:
        if true_mode == pred_mode:
            index_html += f"\n<h2>Correctly classified: {true_mode}</h2>\n<ul>\n"
        else:
            index_html += f"\n<h2>{pair}</h2>\n<ul>\n"
        current_pair = pair
    index_html += f'<li><a href="{fpath.name}">{trip_key}</a></li>\n'

index_html += "</ul>\n</body></html>"

index_path = OUT_DIR / "index.html"
with open(index_path, "w") as f:
    f.write(index_html)

print(f"\n{'='*60}")
print(f"Generated {len(all_maps)} maps in {OUT_DIR}/")
print(f"Open {index_path} in a browser to browse them.")
print(f"{'='*60}")
