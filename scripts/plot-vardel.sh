#!usr/bin/bash
#AdaptDel authors authored this file

configs=( \
config/plot/roberta-base.yelp.all.certified_volume_accuracy.yaml
config/plot/roberta-base.yelp.all.quantile_certified_volume_accuracy.yaml
)

# Loop through each configuration and run the main command
for config in "${configs[@]}"; do
  cmd="pipenv run python main.py --mode plot --config_path $config --override_config"
  $cmd || { echo "Command failed: $cmd"; exit 1; }
done