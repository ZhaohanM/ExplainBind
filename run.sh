#!/bin/bash
# ────────────────────────────────────────────────────────────────
# Run ExplainPLI experiments across multiple dataset splits
# Logs to Weights & Biases (wandb)
# ────────────────────────────────────────────────────────────────

dataset_list=("PDB" "BindingDB" 'Human' 'Biosnap')
split_list=("S0", "25", "28", "31", "33", "006", "043", "062", "088")

# Navigate to ExplainPLI directory
cd ../ExplainPLI/ || exit

for dataset in "${dataset_list[@]}"; do
  for split in "${split_list[@]}"; do
    echo "🚀 Running experiment with --dataset=$dataset --split=$split"
    
    # Run the experiment (rename main_all_S1.py to your actual main file if needed)
    WANDB_RUN_NAME="${dataset}_${split}" \
    python main_all_S1.py --dataset "$dataset" --split "$split"
    
    # Prevent overlapping runs
    sleep 5
  done
done

