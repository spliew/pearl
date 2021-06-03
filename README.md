# PEARL
  - This repository contains source codes to reproduce the results in **PEARL: Data Synthesis with Private Embeddings and Adversarial Reconstruction Learning**.

## Structure
  - `image`: contains the code for image dataset experiment. To run, use `main.py`. To evaluate, use `eval.py`.
  - `tabular`: contains the code for tabular dataset experiment. To run or evaluate, use `run.py`.
  
## Dependencies
Version numbers are based on our machine and may need not to be matched exactly.
```
scikit-learn 0.23.1
pytorch 1.5.0
torchvision 0.6.0
matplotlib 3.2.1
seaborn 0.10.1
sdgym 0.2.2
```
## License
This implementation is licensed under the Apache License 2.0.

## Acknowledgement
Our implementation refers to the source code from the following repositories:
  * [Technical Report: Relational Data Synthesis using Generative Adversarial Networks: A Design Space Exploration](https://github.com/ruclty/Daisy)
  * [DP-MERF](https://github.com/frhrdr/dp-merf)
  * [GAN Metrics Pytorch](https://github.com/abdulfatir/gan-metrics-pytorch); licensed under the Apache License 2.0.
