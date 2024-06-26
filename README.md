# CERT-ED: Certifiably Robust Text Classification for Edit Distance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 Abstract
With the growing integration of AI in daily life, ensuring the robustness of systems to inference-time attacks is crucial. Among the approaches for certifying robustness to such adversarial examples, randomized smoothing has emerged as highly promising due to its nature as a wrapper around arbitrary black-box models. Previous work on randomized smoothing in natural language processing has primarily focused on specific subsets of edit distance operations, such as synonym substitution or word insertion, without exploring the certification of all edit operations. In this paper, we adapt Randomized Deletion (Huang et al., 2023) and propose CERTified Edit Distance defense (CERT-ED) for natural language classification. Through comprehensive experiments, we demonstrate that CERT-ED outperforms the existing Hamming distance method RanMASK (Zeng et al., 2023) in 4 out of 5 datasets in terms of both accuracy and the cardinality of the certificate. By covering various threat models, including 5 direct and 5 transfer attacks, our method improves empirical robustness in 38 out of 50 settings. This work is submitted to ACL 2024 June rolling review.

---

## 📂 Directory Structure

```plaintext
.
├── config
│   ├── attack                        # Sample configurations for attack experiments
│   ├── certify                       # Sample configurations for certification experiments
│   ├── train                         # Sample configurations for training experiments
│   └── plot                          # Sample configurations for plotting experimental results
├── libs                              # External libraries and dependencies
│   └── TextAttack                    # Library for modified TextAttack code
├── outputs                           # Directory for experimental outputs
├── scripts                           # Shell scripts for running various steps
│   ├── attack-roberta.sh             # Sample script for running attacks on the Roberta model
│   ├── certify-roberta.sh            # Sample script for running certification on the Roberta model
│   ├── plot.sh                       # Sample script for plotting the certified results
│   └── train-roberta.sh              # Sample script for training the Roberta model
├── src                               # Source code directory
│   ├── adv_attack                    # Package for adversarial attack implementations
│   │   └── Various attack scripts (e.g., BAE, CLARE, Fast BERT) 
│   ├── certification                 # Package for certification mechanisms and utilities
│   │   └── Various certification scripts (e.g., edit_certs, masking_mech, smoothed_classifier) 
│   ├── attack.py                     # Main script for performing attacks, called by main.py
│   ├── certify.py                    # Main script for the certification process, called by main.py
│   ├── main.py                       # Main entry point for training, certifying, plotting, and attacking
│   ├── train.py                      # Script for training models, called by main.py
│   └── visualization.py              # Script for visualization tasks, called by main.py

```

---

## 🚀 Reproducing Experiments

### Environment Setup

Before running any experiments, set up the virtual environment using Pipenv:

```bash
pipenv install
```

### 1. **Model Training**

- Train the smoothed model via `main.py` with `--mode train`.
- Example: See `scripts/train-roberta.sh`.

```bash
pipenv run python main.py --mode train --config_path config/train/$CONFIG_FILE.yaml --override_config
```

### 2. **Prediction and Certification**

- Save base model confidence scores and compute the certified radius via `main.py` with `--mode certify`.
- Example: See `scripts/certify-roberta.sh`.

```bash
pipenv run python main.py --mode certify --config_path config/certify/$CONFIG_FILE.yaml --override_config
```

### 3. **Adversarial Attack**

- Run adversarial attacks on the model via `main.py` with `--mode attack`.
- Example: See `scripts/attack-roberta.sh`.

```bash
pipenv run python main.py --mode attack --config_path config/attack/$CONFIG_FILE.yaml --override_config
```

### 4. **Visualization**

- Generate certified accuracy visualizations for the results via `main.py` with `--mode plot`.

```bash
pipenv run python main.py --mode plot --config_path config/plot/$CONFIG_FILE.yaml --override_config
```

### Custom Experiments

- You can customize and run your own experiments by defining your own configuration `.yaml` files. Place your custom configuration file in the appropriate `config` subdirectory (`train`, `certify`, `attack`, or `plot`), and use it with the respective command.

```bash
pipenv run python main.py --mode <mode> --config_path config/<subdir>/$YOUR_CUSTOM_CONFIG.yaml --override_config
```

Replace `<mode>` with one of `train`, `certify`, `attack`, or `plot` and `<subdir>` with the corresponding subdirectory (`train`, `certify`, `attack`, `plot`).

## 📊 Datasets

The NLP datasets used in our experiments are automatically downloaded from Hugging Face and the [AdvBench](https://github.com/thunlp/Advbench) repository.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

## 📝 Acknowledgement

We would like to recognize the contributions of AI writing assistants, particularly ChatGPT and GitHub Copilot, in the development of this project. These tools offered valuable suggestions and helped generate initial code frameworks. However, the concepts and structure of the project remain our original work and were not directly shaped by AI-generated content. We are grateful for the role these tools played in accelerating the writing process and improving the overall quality of the project.

Additionally, we acknowledge the inclusion of modified versions of source codes from the following repositories:

- [TextAttack](https://github.com/QData/TextAttack) (for libraries in `libs/TextAttack`)
- [TextCRS](https://github.com/Eyr3/TextCRS) (for the BAE insertion attack recipe)
- [RS-Del](https://github.com/Dovermore/randomized-deletion) (For some Randomized Deletion codes)
- [RanMASK](https://github.com/zjiehang/RanMASK) (For Masking related codes)

We follow their respective licenses in utilizing and modifying their codebases.
