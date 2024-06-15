#!usr/bin/bash
#CERTified Edit Distance defense (CERT-ED) authors authored this file
#ChatGPT and/or Copilot are used in generating scaffolding code for this file

configs=( \
config/train/roberta-base.ag-news.base.yaml
config/train/roberta-base.ag-news.deletion_09.yaml
config/train/roberta-base.ag-news.masking_09.yaml
)

# Loop through each configuration and run the main command
for config in "${configs[@]}"; do
  cmd="pipenv run python main.py --mode train --config_path $config --override_config"
  $cmd || { echo "Command failed: $cmd"; exit 1; }
done