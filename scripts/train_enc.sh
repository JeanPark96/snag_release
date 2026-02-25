#!/usr/bin/env bash

MAX_RETRIES=5
 
names=(charades_sta_i3d_mind_enc_latent)
seeds=(101 303 404)

for seed in "${seeds[@]}"; do
  for name in "${names[@]}"; do
    success=0
    for i in $(seq 1 "$MAX_RETRIES"); do
      echo "=== seed=$seed name=$name Attempt $i/$MAX_RETRIES ==="
      if python ./train.py --seed "$seed" --opt "video_centric/${name}.yaml" --name "${name}_${seed}"; then
        success=1
        break
      fi
      echo "Crashed. Retrying in 10s..."
      sleep 10
    done

    if [[ "$success" -ne 1 ]]; then
      echo "FAILED after $MAX_RETRIES retries: seed=$seed name=$name" >&2
      exit 1   # or: continue  # if you prefer to keep going
    fi
    python ./eval.py --name "${name}_${seed}" --ckpt last
  done
done