# ChatARangAI v2

## Run
./run.sh
./.venv/bin/python agent.py --device 1 --subject S01 --condition interrupt --calib_time 2
./.venv/bin/python aggregate.py

## Keys
c calibrate/start, e end, q quit, space confirm

## Outputs
logs/session_*.csv
logs/plots/session_*_{angle,drift}.png
logs/reports/{trial_summaries,aggregate_summary}.csv

## NMES (pilot)
ch1 quads=UP, ch2 lumbar=DOWN; PW 200µs; ~1.5s; period≥3s; stop |drift|≥12° for ≥2s
