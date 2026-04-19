#!/usr/bin/env bash

MAX_RETRIES=5
 
names=(
        # charades_sta_i3d_default
        # charades_sta_i3d_dec1 
        # charades_sta_i3d_dec3 
        # charades_sta_i3d_dec4
        charades_sta_i3d_dec5 
        charades_sta_i3d_dec6 
        charades_sta_i3d_enc2 
        charades_sta_i3d_enc3
        charades_sta_i3d_enc4
        # charades_sta_i3d_enc2_dec1
        # charades_sta_i3d_enc2_dec3
        # charades_sta_i3d_enc2_dec4
        # charades_sta_i3d_enc3_dec1
        # charades_sta_i3d_enc3_dec3
        # charades_sta_i3d_enc3_dec4
        # charades_sta_i3d_enc4_dec1
        # charades_sta_i3d_enc4_dec3
        # charades_sta_i3d_enc4_dec4
        )
        
seeds=(101 303 404 505 606 707 808 909 1001 1107)

for seed in "${seeds[@]}"; do
  for name in "${names[@]}"; do
    success=0
    for i in $(seq 1 "$MAX_RETRIES"); do
      echo "=== seed=$seed name=$name Attempt $i/$MAX_RETRIES ==="
      if python ./train.py --seed "$seed" --opt "fixed_depth_split/${name}.yaml" --folder fixed_depth_split --name "${name}_${seed}"; then
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

    eval_csv="./experiments_train_val_split/fixed_depth_split/${name}_${seed}/per_sample_eval_last.csv"

    if [[ -f "$eval_csv" ]]; then
      echo "Skipping eval: found $eval_csv"
    else
      echo "Running eval: $eval_csv not found"
      python ./eval.py --name "${name}_${seed}" --ckpt best --folder fixed_depth_split
    fi

  done
done