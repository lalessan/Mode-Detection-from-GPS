"""Plot misclassified trips on interactive maps.

For each major confusion pair, picks example trips and creates a folium
HTML map showing the GPS trace colored by speed.

Usage:
    python map_errors.py
    # Open the HTML files in a browser
"""
import shutil
import numpy as np
import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson
from pathlib import Path

CHECKPOINT_DIR = Path("checkpoints/")
OUT_DIR = Path("error_maps/")

# ── Delete old maps ───────────────────────────────────────────────
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
    print(f"Deleted old maps in {OUT_DIR}/")
OUT_DIR.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────

print("Loading OOF predictions...")
oof = pd.read_parquet(CHECKPOINT_DIR / "fit_steps" / "05_oof_predictions.parquet")
true_col, pred_col = "true_label", "pred_label"

print("Loading GPS data...")
gps = pd.read_parquet(CHECKPOINT_DIR / "fit_steps" / "01_gps_motion.parquet")

# Load stats for feature context
stats = pd.read_parquet(CHECKPOINT_DIR / "fit_steps" / "02_stats.parquet")

# Compute displacement (start-to-end distance) per trip
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

gps_key = "trip_key" if "trip_key" in gps.columns else "KEY"
first_pts = gps.groupby(gps_key).first()[["LATITUDE", "LONGITUDE"]]
last_pts = gps.groupby(gps_key).last()[["LATITUDE", "LONGITUDE"]]
disp_vals = haversine_m(
    first_pts["LATITUDE"].values, first_pts["LONGITUDE"].values,
    last_pts["LATITUDE"].values, last_pts["LONGITUDE"].values,
)
disp_df = pd.DataFrame({"trip_key": first_pts.index, "DISPLACEMENT_M": disp_vals})

# Compute max range (furthest GPS point from centroid) per trip
centroids = gps.groupby(gps_key)[["LATITUDE", "LONGITUDE"]].mean()
gps_c = gps.merge(
    centroids.rename(columns={"LATITUDE": "_clat", "LONGITUDE": "_clon"}),
    left_on=gps_key, right_index=True,
)
gps_c["_dc"] = haversine_m(
    gps_c["LATITUDE"].values, gps_c["LONGITUDE"].values,
    gps_c["_clat"].values, gps_c["_clon"].values,
)
max_range_df = gps_c.groupby(gps_key)["_dc"].max().reset_index()
max_range_df.columns = ["trip_key", "MAX_RANGE_M"]
del gps_c

# Merge predictions with stats
oof_stats = oof.merge(stats, on="trip_key", how="left")
oof_stats = oof_stats.merge(disp_df, on="trip_key", how="left")
oof_stats = oof_stats.merge(max_range_df, on="trip_key", how="left")

# ── Load additional feature checkpoints ──────────────────────────

routes_path = CHECKPOINT_DIR / "fit_steps" / "04_route_features.parquet"
routes = pd.read_parquet(routes_path) if routes_path.exists() else None
if routes is not None:
    oof_stats = oof_stats.merge(routes, on="trip_key", how="left", suffixes=("", "_route"))

stops_path = CHECKPOINT_DIR / "fit_steps" / "03_stop_features.parquet"
stops = pd.read_parquet(stops_path) if stops_path.exists() else None
if stops is not None:
    stop_cols = ["trip_key", "MIN_DIST_METRO", "DURATION_METRO", "MIN_DIST_BUS", "DURATION_BUS"]
    stop_cols = [c for c in stop_cols if c in stops.columns]
    oof_stats = oof_stats.merge(stops[stop_cols], on="trip_key", how="left", suffixes=("", "_stop"))

metro_ep_path = CHECKPOINT_DIR / "fit_steps" / "03a_metro_ep_features.parquet"
metro_ep = pd.read_parquet(metro_ep_path) if metro_ep_path.exists() else None
if metro_ep is not None:
    oof_stats = oof_stats.merge(metro_ep, on="trip_key", how="left", suffixes=("", "_mep"))

# ── Define confusion pairs to visualize ───────────────────────────
# Updated for Run 22a error landscape (top confusions by count):
#   SUBWAY→WALKING (571), WALKING→SUBWAY (544), CAR→BUS (401),
#   CAR→WALKING (394), TRAIN→WALKING (260)

PAIRS = [
    ("SUBWAY", "WALKING", 10),   # #1 confusion: 571 trips, 34.8% of SUBWAY
    ("WALKING", "SUBWAY", 5),    # #2 reverse confusion: 544, 3.4% of WALKING
    ("CAR", "BUS", 5),           # #3: 401, 1.8% of CAR
    ("CAR", "WALKING", 5),       # #4: 394, 1.8% of CAR (post max-range filter)
    ("TRAIN", "WALKING", 5),     # #5: 260, 13.5% of TRAIN
    ("BUS", "CAR", 3),           # also notable
    ("SUBWAY", "BUS", 3),
]

CORRECT_EXAMPLES = ["SUBWAY", "BUS", "CAR", "WALKING", "TRAIN"]


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
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, HERE, Garmin, USGS, NGA",
    )

    # Color by speed: blue=slow, red=fast
    coords = list(zip(lats, lons))

    # Static trace (faded background)
    for i in range(len(coords) - 1):
        speed = speeds[i]
        t = min(speed / 5.0, 1.0)
        r = int(255 * t)
        g = 0
        b = int(255 * (1 - t))
        color = f"#{r:02x}{g:02x}{b:02x}"

        folium.PolyLine(
            [coords[i], coords[i + 1]],
            color=color, weight=2, opacity=0.3,
        ).add_to(m)

    # Animated playback using TimestampedGeoJson
    timestamps = trip_gps["timestamp"].values
    time_strs = pd.to_datetime(timestamps).strftime("%Y-%m-%dT%H:%M:%S").tolist()

    features = []
    for i in range(len(lats)):
        speed = speeds[i]
        t = min(speed / 5.0, 1.0)
        r = int(255 * t)
        b = int(255 * (1 - t))
        color = f"#{r:02x}00{b:02x}"

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(lons[i]), float(lats[i])],
            },
            "properties": {
                "time": time_strs[i],
                "style": {"color": color},
                "icon": "circle",
                "iconstyle": {
                    "fillColor": color,
                    "fillOpacity": 0.8,
                    "stroke": "true",
                    "radius": 6,
                },
                "popup": f"Speed: {speed:.1f} m/s",
            },
        })

    for i in range(len(lats) - 1):
        speed = speeds[i]
        t = min(speed / 5.0, 1.0)
        r = int(255 * t)
        b = int(255 * (1 - t))
        color = f"#{r:02x}00{b:02x}"

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(lons[i]), float(lats[i])],
                    [float(lons[i + 1]), float(lats[i + 1])],
                ],
            },
            "properties": {
                "time": time_strs[i],
                "style": {"color": color, "weight": 4, "opacity": 0.8},
            },
        })

    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period="PT10S",
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=10,
        loop_button=True,
        time_slider_drag_update=True,
    ).add_to(m)

    # Start/end markers
    folium.Marker(coords[0], popup="START", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(coords[-1], popup="END", icon=folium.Icon(color="red")).add_to(m)

    # Info panel
    info = f"True: {true_mode} | Pred: {pred_mode}"
    if stats_row is not None:
        speed_avg = stats_row.get("SPEED_AVG", 0)
        total_time = stats_row.get("TOTAL_TIME", 0)
        gap_max = stats_row.get("GAP_MAX", 0)
        dist = stats_row.get("DIST_TOTAL_WGS84", 0)
        displacement = stats_row.get("DISPLACEMENT_M", 0)
        max_range = stats_row.get("MAX_RANGE_M", 0)
        subway_pct = stats_row.get("subway_match_pct", 0)
        bus_pct = stats_row.get("bus_match_pct", 0)
        metro_start = stats_row.get("METRO_DIST_START_M", float("nan"))
        metro_end = stats_row.get("METRO_DIST_END_M", float("nan"))
        info += (f"<br>Speed avg: {speed_avg:.1f} m/s, max gap: {gap_max:.0f}s"
                 f"<br>Duration: {total_time:.0f}s, Dist: {dist:.0f}m"
                 f"<br>Displacement: {displacement:.0f}m, Max range: {max_range:.0f}m"
                 f"<br>subway_match: {subway_pct:.2f}, bus_match: {bus_pct:.2f}")
        if not np.isnan(metro_start):
            info += f"<br>Metro start: {metro_start:.0f}m, end: {metro_end:.0f}m"

    title_html = f'''
    <div style="position:fixed; top:10px; left:60px; z-index:9999;
                background:white; padding:10px; border:2px solid black;
                font-size:14px; font-family:monospace;">
        <b>{trip_key}</b><br>{info}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    legend_html = '''
    <div style="position:fixed; bottom:30px; left:60px; z-index:9999;
                background:white; padding:8px; border:1px solid grey;
                font-size:12px;">
        <b>Speed</b><br>
        <span style="color:#0000ff;">&#9632;</span> Slow (0 m/s)&nbsp;
        <span style="color:#ff0000;">&#9632;</span> Fast (5+ m/s)
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def save_map(trip_gps, trip_key, true_mode, pred_mode, stats_row, fname, all_maps, label=None):
    """Helper: build map, save, append to index."""
    trip_data = trip_gps[trip_gps[gps_key] == trip_key].sort_values("timestamp")
    if len(trip_data) == 0:
        print(f"    {trip_key}: no GPS data found")
        return
    m = make_trip_map(trip_data, trip_key, true_mode, pred_mode, stats_row)
    fpath = OUT_DIR / fname
    m.save(str(fpath))
    all_maps.append((label or true_mode, pred_mode, trip_key, fpath))
    print(f"    Saved: {fpath}")


# ── Generate maps ─────────────────────────────────────────────────

all_maps = []

# Main confusion pairs — sorted by displacement (mix of stationary and moving)
for true_mode, pred_mode, n_examples in PAIRS:
    mask = (oof_stats[true_col] == true_mode) & (oof_stats[pred_col] == pred_mode)
    confused = oof_stats[mask].sort_values("DISPLACEMENT_M", ascending=True).head(n_examples)

    if len(confused) == 0:
        print(f"  No examples for {true_mode} -> {pred_mode}")
        continue

    print(f"\n  {true_mode} -> {pred_mode}: {len(confused)} examples (sorted by displacement)")

    for _, row in confused.iterrows():
        trip_key = row["trip_key"]
        fname = f"{true_mode}_to_{pred_mode}_{trip_key.replace('/', '_')}.html"
        save_map(gps, trip_key, true_mode, pred_mode, row, fname, all_maps)

# SUBWAY->WALKING with HIGH displacement (the trips that actually moved far)
sub_walk_mask = (oof_stats[true_col] == "SUBWAY") & (oof_stats[pred_col] == "WALKING")
sub_walk_moving = oof_stats[sub_walk_mask].sort_values("DISPLACEMENT_M", ascending=False).head(10)
print(f"\n  SUBWAY->WALK (highest displacement): {len(sub_walk_moving)} examples")
for _, row in sub_walk_moving.iterrows():
    trip_key = row["trip_key"]
    fname = f"SUBWAY_WALK_moving_{trip_key.replace('/', '_')}.html"
    save_map(gps, trip_key, "SUBWAY", "WALKING", row, fname, all_maps,
             label="SUBWAY->WALK (moving)")

# SUBWAY->WALKING near metro (metro_dist_start < 300m) — these SHOULD be catchable
if "METRO_DIST_START_M" in oof_stats.columns:
    sub_near = oof_stats[sub_walk_mask & (oof_stats["METRO_DIST_START_M"] < 300)]
    sub_near = sub_near.sort_values("METRO_DIST_START_M").head(5)
    print(f"\n  SUBWAY->WALK (near metro, start<300m): {len(sub_near)} examples")
    for _, row in sub_near.iterrows():
        trip_key = row["trip_key"]
        fname = f"SUBWAY_WALK_near_metro_{trip_key.replace('/', '_')}.html"
        save_map(gps, trip_key, "SUBWAY", "WALKING", row, fname, all_maps,
                 label="SUBWAY->WALK (near metro)")

# WALKING->SUBWAY with HIGH speed (false subway predictions on fast walkers?)
walk_sub_mask = (oof_stats[true_col] == "WALKING") & (oof_stats[pred_col] == "SUBWAY")
walk_sub_fast = oof_stats[walk_sub_mask].sort_values("SPEED_AVG", ascending=False).head(5)
print(f"\n  WALKING->SUBWAY (highest speed): {len(walk_sub_fast)} examples")
for _, row in walk_sub_fast.iterrows():
    trip_key = row["trip_key"]
    fname = f"WALK_SUBWAY_fast_{trip_key.replace('/', '_')}.html"
    save_map(gps, trip_key, "WALKING", "SUBWAY", row, fname, all_maps,
             label="WALKING->SUBWAY (fast)")

# CAR->BUS: sorted by bus_match_pct to see if these are near bus routes
car_bus_mask = (oof_stats[true_col] == "CAR") & (oof_stats[pred_col] == "BUS")
if "bus_match_pct" in oof_stats.columns:
    car_bus_high = oof_stats[car_bus_mask].sort_values("bus_match_pct", ascending=False).head(5)
    print(f"\n  CAR->BUS (highest bus_match_pct): {len(car_bus_high)} examples")
    for _, row in car_bus_high.iterrows():
        trip_key = row["trip_key"]
        fname = f"CAR_BUS_highpct_{trip_key.replace('/', '_')}.html"
        save_map(gps, trip_key, "CAR", "BUS", row, fname, all_maps,
                 label="CAR->BUS (high bus_pct)")

# TRAIN->WALKING with high displacement (genuine train trips missed)
train_walk_mask = (oof_stats[true_col] == "TRAIN") & (oof_stats[pred_col] == "WALKING")
train_walk_moving = oof_stats[train_walk_mask].sort_values("DISPLACEMENT_M", ascending=False).head(5)
print(f"\n  TRAIN->WALK (highest displacement): {len(train_walk_moving)} examples")
for _, row in train_walk_moving.iterrows():
    trip_key = row["trip_key"]
    fname = f"TRAIN_WALK_moving_{trip_key.replace('/', '_')}.html"
    save_map(gps, trip_key, "TRAIN", "WALKING", row, fname, all_maps,
             label="TRAIN->WALK (moving)")

# Correctly classified examples for comparison
for mode in CORRECT_EXAMPLES:
    mask = (oof_stats[true_col] == mode) & (oof_stats[pred_col] == mode)
    correct = oof_stats[mask].sample(n=min(3, mask.sum()), random_state=42)

    print(f"\n  Correct {mode}: {len(correct)} examples")
    for _, row in correct.iterrows():
        trip_key = row["trip_key"]
        fname = f"CORRECT_{mode}_{trip_key.replace('/', '_')}.html"
        save_map(gps, trip_key, mode, mode, row, fname, all_maps)

# ── Create index page ─────────────────────────────────────────────

index_html = """<html><head>
<title>Error Analysis Maps — Run 22a</title>
<style>
body { font-family: sans-serif; max-width: 900px; margin: 40px auto; }
h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }
h2 { color: #c00; margin-top: 30px; }
h2.correct { color: #060; }
li { margin: 4px 0; }
.stats { color: #666; font-size: 0.9em; }
</style>
</head><body>
<h1>Error Analysis Maps — Run 22a</h1>
<p>Blue = slow, Red = fast. Green marker = start, Red marker = end.<br>
Info panel shows speed, duration, displacement, max range, metro proximity.</p>
"""

current_pair = None
for true_mode, pred_mode, trip_key, fpath in all_maps:
    pair = f"{true_mode} -> {pred_mode}"
    if pair != current_pair:
        if current_pair is not None:
            index_html += "</ul>\n"
        if true_mode == pred_mode:
            index_html += f'\n<h2 class="correct">Correctly classified: {true_mode}</h2>\n<ul>\n'
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
