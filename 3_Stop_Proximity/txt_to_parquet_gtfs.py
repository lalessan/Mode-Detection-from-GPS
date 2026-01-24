import polars as pl
import zipfile
from pathlib import Path
import tempfile

# NOTE: Hardcoded paths -> in the future pass these as args (sys.argv)
# Change the path to desired locations.
FOLDER = Path(r"C:\Users\Mikus\Documents\DTU\Thesis\ThesisGPS\gtfs_idf_2022_2023")
OUT_DIR = Path(r"C:\Users\Mikus\Documents\DTU\Thesis\ThesisGPS\gtfs_parquet")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Get all zip files inside the folder.
zip_files = list(FOLDER.glob("*.zip"))

for zip_path in zip_files:
    print(f"Processing: {zip_path.name}...")

    # This is our new zip title and path.
    out_zip = OUT_DIR / f"{zip_path.stem}_parquet.zip"
    
    # Slightly redundant check
    if out_zip.exists():
        print(f"Skip (exists): {out_zip.name}")
        continue

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        with zipfile.ZipFile(zip_path) as zf:
            # Gather all .txt files inside the archive.
            txt_names = [n for n in zf.namelist() if n.lower().endswith(".txt")]

            for name in txt_names:
                # Extract the CSV to a temp path so scan_csv can read from a file path
                extracted_path = Path(zf.extract(name, path=tmpdir))
                parquet_path = extracted_path.with_suffix(".parquet")
                
                # Using Polars we read the .txt file and then convert it to a parquet file.
                (pl.scan_csv(extracted_path,schema_overrides = {"trip_short_name": pl.Utf8,}).sink_parquet(parquet_path))

                print(f"  {name} -> {parquet_path.name}")

        # Re-zip the generated parquet files from the temp directory.
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as outz:
            for pq in tmpdir.rglob("*.parquet"):
                outz.write(pq, pq.name)

    print(f"  Wrote: {out_zip.name}")
print("Done.")
