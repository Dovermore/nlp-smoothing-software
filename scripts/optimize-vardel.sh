#!usr/bin/bash
#AdaptDel authors authored this file

configs=( \
config/optimize_rate/roberta-base.yelp.var_del_optim.yaml
)

# Loop through each configuration and run the main command
for config in "${configs[@]}"; do
  cmd="pipenv run python main.py --mode optimize_rate --config_path $config --override_config"
  $cmd || { echo "Command failed: $cmd"; exit 1; }
done