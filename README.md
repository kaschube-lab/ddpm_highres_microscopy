# Denoising diffusion models for high-resolution microscopy image restoration

Official code for the paper "Denoising diffusion models for high-resolution microscopy image restoration" by Pamela Osuna-Vargas, Maren H. Wehrheim, Lucas Zinz, Johanna Rahm, Ashwin Balakrishnan, Alexandra Kaminer, Mike Heilemann, and Matthias Kaschube, accepted at WACV 2025.

## Requirements

We recommend creating a new conda environment and installing the following:

`conda install python=3.8.16`

`pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

`pip install pandas numpy tqdm scipy opencv-python`

## How to

To train a new model or use an existing model to do inference, indicate the data and model specifics on a json file as in `config_train/restoration_microtubule_edm2.json`. 

Then:
```python run.py --config config_train/restoration_microtubule_edm2.json --phase train --batch 8 --gpu 0```

If doing inference, make sure to change the phase in the code above, and specify in the json file the checkpoint to be loaded as "resume_state".