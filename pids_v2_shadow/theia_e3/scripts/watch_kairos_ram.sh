#!/usr/bin/env bash
# Sample kairos/TARGET's memory until it plateaus, so the decision to run the SHADOW arm alongside
# it rests on a measurement rather than an analogy. The analogy is what cost us rcaid/TARGET.
# VmHWM is the all-time peak and never decreases; VmRSS sawtooths per batch. Both are recorded.
OUT=/data/pids_v2_shadow/theia_e3/logs/kairos_target_ram.tsv
echo -e "time\tVmRSS_GB\tVmHWM_GB\tMemAvail_GB\tphase" > "$OUT"
while :; do
  p=$(pgrep -f "[p]idsmaker[.]main kairos THEIA_E3_TARGET " | head -1)
  [ -z "$p" ] && { echo -e "$(date +%H:%M:%S)\t-\t-\t-\tPROCESS_GONE" >> "$OUT"; break; }
  rss=$(awk '/^VmRSS:/{printf "%.1f",$2/1048576}' /proc/$p/status 2>/dev/null)
  hwm=$(awk '/^VmHWM:/{printf "%.1f",$2/1048576}' /proc/$p/status 2>/dev/null)
  av=$(awk '/^MemAvailable:/{printf "%.1f",$2/1048576}' /proc/meminfo)
  ph=$(tail -1 /data/pids_v2_shadow/theia_e3/kairos/logs/THEIA_E3_TARGET_e300.log 2>/dev/null | cut -c1-70)
  echo -e "$(date +%H:%M:%S)\t$rss\t$hwm\t$av\t$ph" >> "$OUT"
  sleep 60
done
