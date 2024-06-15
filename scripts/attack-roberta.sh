#!usr/bin/bash
#CERTified Edit Distance defense (CERT-ED) authors authored this file
#ChatGPT and/or Copilot are used in generating scaffolding code for this file

configs=( \
config/attack/roberta-base.ag-news.base.bae-i.yaml
config/attack/roberta-base.ag-news.base.fast_bertattack.yaml
config/attack/roberta-base.ag-news.base.clare_full.yaml
config/attack/roberta-base.ag-news.base.deepwordbug.yaml
config/attack/roberta-base.ag-news.base.textfooler.yaml
config/attack/roberta-base.ag-news.deletion_09.bae-i.yaml
config/attack/roberta-base.ag-news.deletion_09.fast_bertattack.yaml
config/attack/roberta-base.ag-news.deletion_09.clare_full.yaml
config/attack/roberta-base.ag-news.deletion_09.deepwordbug.yaml
config/attack/roberta-base.ag-news.deletion_09.textfooler.yaml
config/attack/roberta-base.ag-news.masking_09.bae-i.yaml
config/attack/roberta-base.ag-news.masking_09.fast_bertattack.yaml
config/attack/roberta-base.ag-news.masking_09.clare_full.yaml
config/attack/roberta-base.ag-news.masking_09.deepwordbug.yaml
config/attack/roberta-base.ag-news.masking_09.textfooler.yaml
)

# Loop through each configuration and run the main command
for config in "${configs[@]}"; do
  cmd="pipenv run python main.py --mode attack --config_path $config --override_config"
  $cmd || { echo "Command failed: $cmd"; exit 1; }
done
