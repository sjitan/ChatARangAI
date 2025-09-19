set -e
/opt/homebrew/bin/python3.11 -m venv .venv || python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir -p logs/plots logs/reports seeds
echo "Run examples:"
echo "./.venv/bin/python agent.py --device 1 --subject LIVE01 --condition interrupt --calib_time 3"
echo "./.venv/bin/python agent.py --seed seeds/seed.mp4 --subject SEED01 --condition control"
