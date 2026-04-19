#!/usr/bin/env bash

MAX_RETRIES=5
 
names=(
        charades_sta_i3d_dec142
        charades_sta_i3d_dec143
        charades_sta_i3d_dec144 
        charades_sta_i3d_dec212
        charades_sta_i3d_dec213
        charades_sta_i3d_dec214
        charades_sta_i3d_dec221
        charades_sta_i3d_dec223
        charades_sta_i3d_dec224 
        charades_sta_i3d_dec312
        charades_sta_i3d_dec313
        charades_sta_i3d_dec314
        charades_sta_i3d_dec321
        charades_sta_i3d_dec322
        charades_sta_i3d_dec323
        charades_sta_i3d_dec324
        charades_sta_i3d_dec331
        charades_sta_i3d_dec332
        charades_sta_i3d_dec334
        charades_sta_i3d_dec412
        charades_sta_i3d_dec413
        charades_sta_i3d_dec414 
        charades_sta_i3d_dec422
        charades_sta_i3d_dec423
        charades_sta_i3d_dec424 
        charades_sta_i3d_dec432
        charades_sta_i3d_dec433
        charades_sta_i3d_dec434
        charades_sta_i3d_dec442
        charades_sta_i3d_dec443
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