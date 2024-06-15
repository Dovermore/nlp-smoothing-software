#!usr/bin/bash
#CERTified Edit Distance defense (CERT-ED) authors authored this file
#ChatGPT and/or Copilot are used in generating scaffolding code for this file

configs=( \
config/plot/roberta-base.ag-news.all.certified_volume_accuracy.yaml
)

# Loop through each configuration and run the main command
for config in "${configs[@]}"; do
  cmd="pipenv run python main.py --mode plot --config_path $config --override_config"
  $cmd || { echo "Command failed: $cmd"; exit 1; }
done