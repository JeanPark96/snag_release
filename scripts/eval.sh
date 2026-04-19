#!/usr/bin/env bash

MAX_RETRIES=5
 
names=(
  #charades_sta_i3d_default
  #      charades_sta_i3d_dec1 
  #      charades_sta_i3d_dec3 
        charades_sta_i3d_dec4 
        )
        
seeds=(101 303 404 505 606 707 808 909 1001 1107)
splits=(train test val)
for seed in "${seeds[@]}"; do
  for name in "${names[@]}"; do
    for split in "${splits[@]}"; do
      success=0
      for i in $(seq 1 "$MAX_RETRIES"); do
        echo "=== seed=$seed name=$name split=$split Attempt $i/$MAX_RETRIES ==="
        if python ./eval.py --ckpt best --folder fixed_depth_split --name "${name}_${seed}" --split "$split" --return_features; then
          success=1
        break
      fi
      echo "Crashed. Retrying in 10s..."
      sleep 10
    done
  done
done
done