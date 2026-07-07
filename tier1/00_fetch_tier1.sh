#!/usr/bin/env bash
# Tier 1 - Step 1 (data): fetch the external datasets the HOLC boundary RD needs.
# Saves under data/tier1/.  Resumable: skips files already present.
#
#   ./tier1/00_fetch_tier1.sh            # everything except the 2.7 GB CHRONEX zip
#   ./tier1/00_fetch_tier1.sh --chronex  # also download+extract CHRONEX-US (~2.7 GB)
#
# URLs verified 2026-07-04 (see tier1/README.md for provenance).
set -u
cd "$(dirname "$0")/.."
T1=data/tier1
mkdir -p "$T1/holc" "$T1/markley" "$T1/hisdac" "$T1/chronex"
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

get() {  # url dest
  local url="$1" dst="$2"
  if [ -s "$dst" ]; then LOG "skip (have) $dst"; return 0; fi
  LOG "GET $url"
  curl -fL --retry 3 -o "$dst.part" "$url" && mv "$dst.part" "$dst" \
    && LOG "  -> $dst ($(du -h "$dst" | cut -f1))" || { LOG "  FAILED $url"; return 1; }
}

# 1) HOLC national polygons (Mapping Inequality) — the RD frontier source.
get "https://dsl.richmond.edu/panorama/redlining/static/mappinginequality.gpkg" \
    "$T1/holc/mappinginequality.gpkg"

# 2) Markley HOLC ADS covariates (OSF qytj8) — pre-treatment balance / covariates.
#    Zip -> DATA_DOWNLOAD/TABLES/ADS_FINAL.csv (join on CITY + HOLC_ID ~ polygon label).
if get "https://osf.io/download/92phs/" "$T1/markley/DATA_DOWNLOAD.zip"; then
  if [ ! -s "$T1/markley/ADS_FINAL.csv" ]; then
    unzip -o -j "$T1/markley/DATA_DOWNLOAD.zip" "*/ADS_FINAL.csv" -d "$T1/markley" >/dev/null 2>&1 \
      && LOG "  extracted ADS_FINAL.csv" || LOG "  (ADS_FINAL.csv not found in zip; inspect manually)"
  fi
fi

# 3) HISDAC-US First Built-Up Year raster (Harvard Dataverse DOI 10.7910/DVN/HHFM5E).
#    EPSG:5070, 250 m; pixel = first built-up year. Threshold <=1940 for a 1940 mask.
get "https://dataverse.harvard.edu/api/access/datafile/7337822" "$T1/hisdac/FBUY.tif"
get "https://dataverse.harvard.edu/api/access/datafile/8165708" "$T1/hisdac/FBUY_README.txt"

# 4) CHRONEX-US dated road network (Figshare 28644674) — 693 per-CBSA GPKGs, ~2.7 GB.
#    Only fetched with --chronex because of size.  After extraction we keep just the
#    six Tier 1 metro GPKGs and delete the zip to save space.
if [ "${1:-}" = "--chronex" ]; then
  get "https://ndownloader.figshare.com/files/53340398" "$T1/chronex/CHRONEX_US_V1.zip"
  if [ -s "$T1/chronex/CHRONEX_US_V1.zip" ]; then
    LOG "CHRONEX zip contents (grep our 6 CBSAs):"
    unzip -l "$T1/chronex/CHRONEX_US_V1.zip" | grep -E "16980|37980|12580|19820|12060|31080" || true
    for CBSA in 16980 37980 12580 19820 12060 31080; do
      f="chronex_us_${CBSA}.gpkg"
      [ -s "$T1/chronex/$f" ] && { LOG "skip (have) $f"; continue; }
      unzip -o -j "$T1/chronex/CHRONEX_US_V1.zip" "*${CBSA}*.gpkg" -d "$T1/chronex" >/dev/null 2>&1 \
        && LOG "  extracted $f" || LOG "  MISSING $f in zip (verify CBSA id)"
    done
  fi
else
  LOG "CHRONEX-US skipped (pass --chronex to download the 2.7 GB dated-network zip)"
fi

LOG "TIER 1 FETCH DONE"
