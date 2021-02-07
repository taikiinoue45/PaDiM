<p align="left">
  <img src=assets/logo.svg width="70%" />
</p>

![black](https://github.com/taikiinoue45/PaDiM/workflows/black/badge.svg)
![blackdoc](https://github.com/taikiinoue45/PaDiM/workflows/blackdoc/badge.svg)
![flake8](https://github.com/taikiinoue45/PaDiM/workflows/flake8/badge.svg)
![isort](https://github.com/taikiinoue45/PaDiM/workflows/isort/badge.svg)
![mypy](https://github.com/taikiinoue45/PaDiM/workflows/mypy/badge.svg)

PyTorch re-implementation of [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://arxiv.org/abs/2011.08785)

<br>

## 1. AUROC Scores

| category   | Paper | My Implementation |
| :-         | :-    | :-                |
| zipper     | 0.985 | 0.923             |
| wood       | 0.949 | 0.992             |
| transistor | 0.975 | 0.998             |
| toothbrush | 0.988 | 0.883             |
| tile       | 0.941 | 0.994             |
| screw      | 0.985 | 0.815             |
| pill       | 0.957 | 0.958             |
| metal_nut  | 0.972 | 0.992             |
| leather    | 0.992 | 1.000             |
| hazelnut   | 0.982 | 0.985             |
| grid       | 0.973 | 0.959             |
| carpet     | 0.991 | 0.997             |
| capsule    | 0.985 | 0.937             |
| cable      | 0.967 | 0.930             |
| bottle     | 0.983 | 1.000             |

<br>

## 2. PRO Scores

| category   | Paper | My Implementation |
| :-         | :-    | :-                |
| zipper     | 0.959 | 0.935             |
| wood       | 0.911 | 0.891             |
| transistor | 0.845 | 0.949             |
| toothbrush | 0.931 | 0.915             |
| tile       | 0.860 | 0.826             |
| screw      | 0.944 | 0.936             |
| pill       | 0.927 | 0.952             |
| metal_nut  | 0.856 | 0.933             |
| leather    | 0.978 | 0.978             |
| hazelnut   | 0.926 | 0.937             |
| grid       | 0.946 | 0.866             |
| carpet     | 0.962 | 0.952             |
| capsule    | 0.935 | 0.921             |
| cable      | 0.888 | 0.918             |
| bottle     | 0.948 | 0.951             |

<br>

## 3. Graphical Results

### zipper
<p align="left">
  <img src=assets/zipper.gif width="100%" />
</p>

### wood
<p align="left">
  <img src=assets/wood.gif width="100%" />
</p>

### transistor
<p align="left">
  <img src=assets/transistor.gif width="100%" />
</p>

### toothbrush
<p align="left">
  <img src=assets/toothbrush.gif width="100%" />
</p>

### tile
<p align="left">
  <img src=assets/tile.gif width="100%" />
</p>

### screw
<p align="left">
  <img src=assets/screw.gif width="100%" />
</p>

### pill
<p align="left">
  <img src=assets/pill.gif width="100%" />
</p>

### metal_nut
<p align="left">
  <img src=assets/metal_nut.gif width="100%" />
</p>

### leather
<p align="left">
  <img src=assets/leather.gif width="100%" />
</p>

### hazelnut
<p align="left">
  <img src=assets/hazelnut.gif width="100%" />
</p>

### grid
<p align="left">
  <img src=assets/grid.gif width="100%" />
</p>

### carpet
<p align="left">
  <img src=assets/carpet.gif width="100%" />
</p>

### capsule
<p align="left">
  <img src=assets/capsule.gif width="100%" />
</p>

### cable
<p align="left">
  <img src=assets/cable.gif width="100%" />
</p>

### bottle
<p align="left">
  <img src=assets/bottle.gif width="100%" />
</p>

<br>

## 4. Requirements
- CUDA 10.2
- nvidia-docker2

<br>

## 5. Usage

a) Download docker image and run docker container
```
docker pull taikiinoue45/mvtec:padim
docker run --runtime nvidia -it --workdir /app --network host taikiinoue45/mvtec:padim /usr/bin/zsh
```

b) Download this repository
```
git clone https://github.com/taikiinoue45/PaDiM.git
cd /app/PaDiM/padim
```

c) Run experiments
```
sh run.sh
```

d) Visualize experiments
```
mlflow ui
```


<br>

## 6. Contacts

- github: https://github.com/taikiinoue45/
- twitter: https://twitter.com/taikiinoue45/
- linkedin: https://www.linkedin.com/in/taikiinoue45/
