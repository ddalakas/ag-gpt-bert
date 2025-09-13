#!/bin/bash

ratio="Ratio_2_4"  # Specify the ratio variant (e.g., Ratio_2_4 for a 50%:50% CLM-MLM model)
task="xpos"

rm -f "models/${task}/${ratio}-${task}-accuracy.txt"

start=$(date +%s)

# Execute the script with three different random seeds
for seed in 42 1 2; do
    echo "Running with seed $seed..."
    python "${task}-tagging.py" "${task}-config.py" "$seed"
done

end=$(date +%s)
runtime=$(((end-start)/60))
echo "Total runtime: $runtime minutes."