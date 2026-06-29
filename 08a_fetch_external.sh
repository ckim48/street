#!/usr/bin/env bash
# Stage 8a: fetch external socioeconomic datasets (the key-free ones).
# Saves into data/external/. Resumable: skips files already present.
set -u
cd "$(dirname "$0")"
EXT=data/external
mkdir -p "$EXT/fars" "$EXT/lodes"
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

get() {  # url dest
  local url="$1" dst="$2"
  if [ -s "$dst" ]; then LOG "skip (have) $dst"; return; fi
  LOG "GET $url"
  curl -fsSL --retry 3 -o "$dst.part" "$url" && mv "$dst.part" "$dst" \
    && LOG "  -> $dst ($(du -h "$dst" | cut -f1))" || LOG "  FAILED $url"
}

# 1) Opportunity Atlas (tract mobility) — small 'simple' file + covariates
get "https://opportunityinsights.org/wp-content/uploads/2018/10/tract_outcomes_simple.csv" "$EXT/oa_tract_outcomes_simple.csv"
get "https://opportunityinsights.org/wp-content/uploads/2018/10/tract_covariates.csv"       "$EXT/oa_tract_covariates.csv"

# 2) Eviction Lab (tract, 2000-2018)
get "https://eviction-lab-data-downloads.s3.amazonaws.com/data-for-analysis/tract_proprietary_valid_2000_2018_y2024m12.csv" "$EXT/evictionlab_tract_2000_2018.csv"

# 3) FARS — pool several years (pedestrian fatalities are rare per tract)
for Y in 2017 2018 2019 2020 2021; do
  get "https://static.nhtsa.gov/nhtsa/downloads/FARS/${Y}/National/FARS${Y}NationalCSV.zip" "$EXT/fars/FARS${Y}.zip"
done

# 4) LODES8 WAC (jobs) — all states + DC, 2021
ST="al ak az ar ca co ct de dc fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy"
for s in $ST; do
  get "https://lehd.ces.census.gov/data/lodes/LODES8/${s}/wac/${s}_wac_S000_JT00_2021.csv.gz" "$EXT/lodes/${s}_wac_2021.csv.gz"
done

LOG "ALL FETCH DONE"
