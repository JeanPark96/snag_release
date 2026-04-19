#!/usr/bin/env bash

MAX_RETRIES=5
 
names=(
        charades_sta_i3d_dec112 
        charades_sta_i3d_dec113 
        charades_sta_i3d_dec114 
        charades_sta_i3d_dec121 
        charades_sta_i3d_dec122 
        charades_sta_i3d_dec123 
        charades_sta_i3d_dec124 
        charades_sta_i3d_dec131 
        charades_sta_i3d_dec132 
        charades_sta_i3d_dec133 
        charades_sta_i3d_dec134 
        charades_sta_i3d_dec141 
        charades_sta_i3d_dec211 
        charades_sta_i3d_dec311 
        charades_sta_i3d_dec411 
        charades_sta_i3d_dec421 
        charades_sta_i3d_dec431 
        charades_sta_i3d_dec441
        )
        
seeds=(101 303 404 505 606)
splits=(test)

for seed in "${seeds[@]}"; do
  for name in "${names[@]}"; do
    success=0
    for i in $(seq 1 "$MAX_RETRIES"); do
      echo "=== seed=$seed name=$name Attempt $i/$MAX_RETRIES ==="
      if python ./train.py --seed "$seed" --opt "fixed_depth_split_branch/${name}.yaml" --folder fixed_depth_split_branch --name "${name}_${seed}"; then
        success=1
        break
      fi
      echo "Crashed. Retrying in 10s..."
      sleep 10
    done

    if [[ "$success" -ne 1 ]]; then
      echo "FAILED after $MAX_RETRIES retries: seed=$seed name=$name" >&2
      exit 1   # or: continue
    fi

    for split in "${splits[@]}"; do
      python ./eval.py --ckpt best --folder fixed_depth_split_branch --name "${name}_${seed}" --split "$split"
    done

  done
done