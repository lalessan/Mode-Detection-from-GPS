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
from folium.plugins import TimestampedGeoJson
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

# Compute displacement (start-to-end distance) per trip
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

first_pts = gps.groupby("trip_key").first()[["LATITUDE", "LONGITUDE"]]
last_pts = gps.groupby("trip_key").last()[["LATITUDE", "LONGITUDE"]]
disp_vals = haversine_m(
    first_pts["LATITUDE"].values, first_pts["LONGITUDE"].values,
    last_pts["LATITUDE"].values, last_pts["LONGITUDE"].values,
)
disp_df = pd.DataFrame({"trip_key": first_pts.index, "DISPLACEMENT_M": disp_vals})

# Merge predictions with stats
oof_stats = oof.merge(stats, on="trip_key", how="left")
oof_stats = oof_stats.merge(disp_df, on="trip_key", how="left")

# ── Define confusion pairs to visualize ───────────────────────────

PAIRS = [
    ("CAR", "WALKING", 15),     # main focus — are these stationary?
    ("SUBWAY", "WALKING", 5),
    ("SUBWAY", "BUS", 3),
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
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, HERE, Garmin, USGS, NGA",
    )

    # Color by speed: blue=slow, red=fast
    max_speed = max(speeds.max(), 1)
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
    # Convert to ISO strings for the plugin
    time_strs = pd.to_datetime(timestamps).strftime("%Y-%m-%dT%H:%M:%S").tolist()

    # Build GeoJSON features: one point per GPS fix (animated marker)
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

    # Add animated line segments too (trail builds up over time)
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
        period="PT10S",         # step every 10 seconds of trip time
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

    # Title
    info = f"True: {true_mode} | Pred: {pred_mode}"
    if stats_row is not None:
        speed_avg = stats_row.get("SPEED_AVG", 0)
        total_time = stats_row.get("TOTAL_TIME", 0)
        gap_max = stats_row.get("GAP_MAX", 0)
        dist = stats_row.get("DIST_TOTAL_WGS84", 0)
        subway_pct = stats_row.get("subway_match_pct", 0)
        bus_pct = stats_row.get("bus_match_pct", 0)
        displacement = stats_row.get("DISPLACEMENT_M", 0)
        info += (f"<br>Speed avg: {speed_avg:.1f} m/s, max gap: {gap_max:.0f}s"
                 f"<br>Duration: {total_time:.0f}s, Dist: {dist:.0f}m, Displacement: {displacement:.0f}m"
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
    # Sort by displacement so we see a range from most stationary to moving
    confused = oof_stats[mask].sort_values("DISPLACEMENT_M", ascending=True).head(n_examples)

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
