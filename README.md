
[[Paper](https://dl.acm.org/doi/abs/10.1145/3748636.3762774)], [[Slides](https://docs.google.com/presentation/d/1SWRFocADGinYXm8KlCutM6XBdRi7ixCf/edit?usp=sharing&ouid=118082991536686796649&rtpof=true&sd=true)], [[Poster](https://docs.google.com/presentation/d/1rsSfUEnVAO33d_ORSlKkbq3XmaiF16CCl9KSVZ0upG4/edit?usp=sharing)]

# Dataset
Download NUMOSIM dataset from https://osf.io/sjyfr/ and add the contents to `data/NUMOSIM/`.

# Requirements
Try to use a GPU with more than 24 GB VRAM for training ideally. 

Use conda to create a virtual environment and pip to install the requirements:

# Environment Setup

```
conda create --name icad python==3.10.13
conda activate icad
pip install -r requirements.txt
```

# Preprocess
Run `python -m utils.preprocess` inside ICAD directory. 

# Training Process
Run `python main.py --task next_prediction` inside ICAD directory. The backbone of the code is based on [TrajGPT implementation](https://github.com/ktxlh/TrajGPT/tree/master/modules). 


# References
```
@inproceedings{azarijoo2025icad,
  author    = {Bita Azarijoo and Maria Despoina Siampou and John Krumm and Cyrus Shahabi},
  title     = {ICAD: A Self-Supervised Autoregressive Approach for Multi-Context Anomaly Detection in Human Mobility Data},
  booktitle = {Proceedings of the 33rd ACM International Conference on Advances in Geographic Information Systems},
  year      = {2025},
}
```