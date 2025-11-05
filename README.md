<div align="center">
<h1> Video Plug-and-Play Optimization for Time-Resolved Computational Microscopy </h1>

**[Yunhui Gao](https://github.com/Yunhui-Gao)** and **[Liangcai Cao](https://scholar.google.com/citations?user=FYYb_-wAAAAJ&hl=en)**

:school: ***[HoloLab](http://www.holoddd.com/)**, Tsinghua University*


:scroll: **Publication Page** **|** :microscope: [**Experimental Dataset**](https://doi.org/10.5281/zenodo.17523561) **|** :key: [**Pretrained Models**](https://github.com/THUHoloLab/ViDNet) **|**  :dart: [**Selective Results**](#-Selective-results)

<p align="center">
<img src="imgs/principle.png", width='800'>
</p>
</div>

Computational microscopy combines advances in optical hardware and signal processing to push the boundaries of imaging resolution and functionality. However, acquiring extended information often comes at the expense of temporal resolution. Here, we present a model-based deep learning framework for **time-resolved imaging in multi-shot computational microscopy**. Building upon the plug-and-play (PnP) optimization theory, our approach integrates the low-level spatiotemporal priors learned from large-scale video datasets with the physical model of an optimized measurement scheme, enabling accurate, time-resolved reconstruction of dynamic scenes. Using lensless ptychographic microscopy as an example, we experimentally demonstrate high-speed holographic imaging of an order of magnitude faster sample dynamics without compromising quality. Additionally, we show that the proposed framework enables high-throughput, label-free imaging of various biological activities of freely moving organisms, such as paramecia and rotifers, with a sensor-limited space-bandwidth-time product of 227 megapixels per second. The presented approach provides a promising solution to time-resolved computational microscopy across a broad range of imaging modalities.


## News


- **2025.07.09** &nbsp; :fire: Simulation code released.

- **2025.07.09** &nbsp; :fire: Pretrained models are released. Click [**here**](https://github.com/THUHoloLab/ViDNet) for more details.


## Requirements

Deep PnP algorithms are implemented with Python in Spyder. Experimental pre- and post-processing codes are written in MATLAB.

- MATLAB R2022b or newer versions
- Python 3.9, PyTorch >= 2.3.1
- Platforms: Windows 10 / 11

## Quick Examples

##### 1. Prepare the environment

- Download the necessary packages according to [`requirements.txt`](https://github.com/THUHoloLab/STRIVER-deep/blob/master/requirement.txt).

##### 2. Download the pretrained models

- Download the pretrained models for ViDNet and the baseline networks, which can be found [**here**](https://github.com/THUHoloLab/ViDNet/releases). Then, move the `.pth` files into the corresponding folders in [`models`](https://github.com/THUHoloLab/STRIVER-deep/blob/master/models/). Note that for the baseline networks, FastDVDnet has been modified for grayscale video denoising and with batch normalization layers removed. DRUNet has adopted the original architecture and pretrained model provided by the authors (click [**here**](https://github.com/cszn/DPIR/tree/master) for more details).

##### 3. Download the simulation and experimental dataset

- Follow the instructions [**here**](https://github.com/THUHoloLab/STRIVER-deep/blob/master/data/README.md) to download and prepare the dataset.

##### 4. Run demo codes

- **Quick demonstration with simulated data.** Run [`demo_sim.py`](https://github.com/THUHoloLab/STRIVER-deep/blob/master/demo_sim.py) with default parameters.
- **Demonstration with experimental data.** First run [`demo_exp_probe_recovery.m`](https://github.com/THUHoloLab/STRIVER-deep/blob/master/demo_exp_probe_recovery.m) for TV-regularized blind ptychographic reconstruction to retrieve the probe profile and an initial estimate of the sample field. Then run [`demo_exp.py`](https://github.com/THUHoloLab/STRIVER-deep/blob/master/demo_exp.py) for the deep PnP reconstruction.
- **Experimental comparison.** Run `demo_exp_comparison_eXgY.py` with default parameters, where `X` and `Y` denote the experiment and group indices of the dataset.



## Selective Results

##### 1. Holographic imaging of freely moving organisms

We experimentally demonstrate time-resolved holographic imaging of freely moving organisms based on coded ptychography. The following results show the holographic videos of paramecium and rotifer samples, visualized in the HSL color space.

<p align="left">
<img src="imgs/paramecia_1.gif", width="394"> &nbsp;
<img src="imgs/paramecia_2.gif", width="394">
<p align="left">

<p align="left">
<img src="imgs/rotifers.gif", width="800">
<p align="left">


##### 2. Quantitative comparison with existing PnP priors

The following figure shows the experimental reconstruction of a xylophyta dicotyledon stem section translating at a speed of 5 pixels per frame. Compared with other popular PnP priors including [3DTV](https://github.com/THUHoloLab/STRIVER), [DRUNet](https://github.com/cszn/DPIR), and [FastDVDnet](https://github.com/m-tassano/fastdvdnet), ViDNet maintains the finest spatial textures and the best temporal consistency.

<p align="left">
<img src="imgs/comparison.gif", width='800'>
</p>

The following table summarizes the average amplitude PSNR (dB) under varying sample translation speeds. ViDNet yields competitive performance even when the sample is moving **almost an order of magnitude faster**! :wink:

| Speed (pixel/frame) | 3DTV              | DRUNet + 3DTV | FastDVDnet + 3DTV | ViDNet + 3DTV     |
| :----:              | :----:            | :----:        | :----:            | :----:            |
| 0                   | **21.17 (+0.30)** | 17.74         | 18.26             | 20.87             |
| 1                   | 15.68             | 15.58         | 16.87             | **19.83 (+2.96)** |
| 2                   | 14.56             | 14.58         | 16.01             | **19.33 (+3.32)** |
| 3                   | 14.09             | 14.27         | 15.59             | **19.18 (+3.59)** |
| 4                   | 13.80             | 14.07         | 15.05             | **18.77 (+3.72)** |
| 5                   | 13.61             | 13.93         | 14.49             | **18.44 (+3.95)** |
| 6                   | 13.47             | 13.84         | 14.00             | **17.96 (+3.96)** |
| 7                   | 13.37             | 13.77         | 13.64             | **17.40 (+3.63)** |
| 8                   | 13.30             | 13.65         | 13.37             | **17.00 (+3.35)** |
| 9                   | 13.24             | 13.63         | 13.18             | **16.55 (+2.92)** |


## Citation

```BibTex
@article{gao2025model,
  title={Model-based deep learning enables time-resolved computational microscopy},
  author={Gao, Yunhui and Cao, Liangcai},
  journal={xxxx},
  volume={xxxx},
  number={xxxx},
  pages={xxxx},
  year={2025},
  publisher={xxxx}
}
```