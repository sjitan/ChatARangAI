# ChatARangAI v2
## Quickstart
./run.sh
./.venv/bin/python agent.py --device 1 --subject LIVE01 --condition interrupt --calib_time 3
## Keys
c=calibrate/start, e=end, q=quit, space=confirm shock
## Outputs
- logs/session_*.csv
- logs/plots/session_*_angle.png
- logs/reports/trial_summaries.csv
- logs/reports/aggregate_summary.csv
- logs/plots/aggregate_slope_scatter.png, aggregate_ttf_hist.png
