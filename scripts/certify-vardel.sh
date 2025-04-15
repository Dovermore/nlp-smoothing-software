#!usr/bin/bash
#AdaptDel authors authored this file

configs=( \
config/certify/roberta-base.yelp.base.yaml
config/certify/roberta-base.yelp.deletion_09.yaml
config/certify/roberta-base.yelp.masking_09.yaml
config/certify/roberta-base.yelp.var_del_len_13_090.yaml
config/certify/roberta-base.yelp.var_del_optim.yaml
)

# Loop through each configuration and run the main command
for config in "${configs[@]}"; do
  cmd="pipenv run python main.py --mode certify --config_path $config --override_config"
  $cmd || { echo "Command failed: $cmd"; exit 1; }
done