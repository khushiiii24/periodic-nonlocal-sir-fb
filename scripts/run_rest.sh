#!/bin/bash
cd "$(dirname "$0")"
for s in s01_thresholds s02_convergence s03_speed_mu s04_mustar s06_full_sir s06b_full_diagnostics s07_sensitivity; do
  echo "== $s $(date +%T)"; python3 $s.py > ../results/$s.log 2>&1; echo "done $s $(date +%T)"
done
