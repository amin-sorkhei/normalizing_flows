# Normalizing Flows (GLOW)

PyTorch implementation for [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039).

Following is samples at different temperatures from a model trained on CelebA dataset with 3 layers (L=3) each consisting of 32 steps (K=32) after training for 100 epcohs .

![](https://github.com/amin-sorkhei/normalizing_flows/blob/master/demo/temperatures.gif)


[link to the trained model](https://drive.google.com/file/d/1zS520AcBaTPJ8r3Wx29qqM2k3j9S-pJi/view?usp=sharing)


# How to Run the Code

Create a virtual environment
```
python3 -m pip venev normalizing_flows_env
```
activate the above environment and install the package by (use `-e` if you are planning to develop new code)
```
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## training from scratch
If you want to train a model from scratch, change the configs and start the job.
```
python3 ./normalizing_flows/train.py --task train --config_path ./configs/[test.yaml | medium.yaml] --device cuda:0
```

## resume training 
If you want to resume training from a model checkpoint. Keep in mind: 
1. you need to provide the seed in order to keep the sampling procedure comparable with the previous run
2. you need to provide the path to the checkpoint and sampling directoies in the config files

```
python3 ./normalizing_flows/train.py --task resume_training --config_path ./configs/[test.yaml | medium.yaml] --device cuda:0
```
