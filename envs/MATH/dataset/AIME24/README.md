---
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: problem
    dtype: string
  - name: solution
    dtype: string
  - name: answer
    dtype: string
  - name: url
    dtype: string
  - name: year
    dtype: string
  splits:
  - name: train
    num_bytes: 139586
    num_examples: 30
  download_size: 81670
  dataset_size: 139586
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Dataset card for AIME 2024

This dataset consists of 30 problems from the 2024 [AIME I](https://artofproblemsolving.com/wiki/index.php/2024_AIME_I?srsltid=AfmBOoqP9aelPNCpuFLO2bLyoG9_elEBPgqcYyZAj8LtiywUeG5HUVfF) and [AIME II](https://artofproblemsolving.com/wiki/index.php/2024_AIME_II_Problems/Problem_15) tests. The original source is [AI-MO/aimo-validation-aime](https://huggingface.co/datasets/AI-MO/aimo-validation-aime), which contains a larger set of 90 problems from AIME 2022-2024.