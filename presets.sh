#!/usr/bin/env bash
set -e
V=.venv/bin/python
S="$1"; C="$2"; IDX="$3"; SEED="$4"
case "$C" in
  control)
    ARGS="--calib_time 4 --warmup 1.0 --ema_alpha 0.06 --need_drift 999 --need_hold 9.9 --hyst_reset 9 --period 5 --safety_drift 26 --safety_hold 4.0"
    ;;
  static)
    ARGS="--calib_time 4 --warmup 1.0 --ema_alpha 0.06 --need_drift 12 --need_hold 2.2 --hyst_reset 5 --period 6 --static_period 12 --safety_drift 24 --safety_hold 3.8"
    ;;
  interrupt|*)
    ARGS="--calib_time 4 --warmup 1.0 --ema_alpha 0.06 --need_drift 10 --need_hold 2.0 --hyst_reset 4 --period 5 --safety_drift 22 --safety_hold 3.5"
    ;;
esac
if [ -n "$SEED" ]; then
  $V agent.py --seed "$SEED" --subject "$S" --condition "$C" $ARGS
else
  $V agent.py --device "$IDX" --subject "$S" --condition "$C" $ARGS
fi
