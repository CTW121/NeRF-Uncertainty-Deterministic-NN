# NeRF uncertainty with Deterministic Neural Network approach

This implementation is built upon PyTorch re-implementation of [nerf-pytorch](https://github.com/krrish94/nerf-pytorch).

## What is a NeRF?

### [Project](http://tancik.com/nerf) | [Video](https://youtu.be/JuH79E8rdKc) | [Paper](https://arxiv.org/abs/2003.08934)

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://tancik.com/nerf)  
 [Ben Mildenhall](https://people.eecs.berkeley.edu/~bmild/)\*<sup>1</sup>,
 [Pratul P. Srinivasan](https://people.eecs.berkeley.edu/~pratul/)\*<sup>1</sup>,
 [Matthew Tancik](http://tancik.com/)\*<sup>1</sup>,
 [Jonathan T. Barron](http://jonbarron.info/)<sup>2</sup>,
 [Ravi Ramamoorthi](http://cseweb.ucsd.edu/~ravir/)<sup>3</sup>,
 [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html)<sup>1</sup> <br>
 <sup>1</sup>UC Berkeley, <sup>2</sup>Google Research, <sup>3</sup>UC San Diego  
  \*denotes equal contribution

A neural radiance field (NeRF) is a simple fully connected network (weights are ~5MB) trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views.

Optimizing a NeRF takes between a few hours and a day or two (depending on resolution) and only requires a single GPU. Rendering an image from an optimized NeRF takes somewhere between less than a second and ~30 seconds, again depending on resolution.

## NeRF Uncertainty Deterministic Neural Network

This NeRF Uncertainty Deterministic Neural Network (NN) estimates the uncertainty of the model’s prediction, i.e., the output estimated uncertainty by the model represents the capability and confidence of the model to synthesize the scene accurately. It outputs an uncertainty prediction based on a single forward pass within a deterministic network. The deterministic network, represented as a multilayer perceptron (MLP) estimating color $\boldsymbol{c}$, volume density $\sigma$, and an uncertainty $\delta$. We adopt the uncertainty estimation method from Recursive Neural Radiance ([Recursive-NeRF](https://ieeexplore.ieee.org/document/9909994)).

Following figure illustrates the NeRF Uncertainty Deterministic NN model pipeline.
![NeRF Uncertainty Deterministic NN pipeline](https://github.com/CTW121/NeRF-Uncertainty-Deterministic-NN/blob/master/images/Uncertainty_Neural_Network_pipeline.png)

After training the model, the uncertainty estimated by the model can be visualized using our uncertainty visualization tool, see section [Visualization uncertainty in NeRF Uncertainty Deterministic NN](##Visualization-uncertainty-in-NeRF-Uncertainty-Deterministic-NN) for more information. 

## How to train NeRF Uncertainty Deterministic Neural Network model

### Run training

First edit `config/chair.yml` to specify your own parameters.

The training script can be invoked by running
```bash
python train_nerf.py --config config/chair.yml
```

### Optional: Resume training from a checkpoint

Resume training from a previous checkpoint, run
```bash
python train_nerf.py --config config/chair.yml --load-checkpoint path/to/checkpoint.ckpt
```

Refer to [nerf-pytorch](https://github.com/krrish94/nerf-pytorch) for the detail of implementation and model training.

## Visualization uncertainty in NeRF Uncertainty Deterministic NN

The uncertainty estimated by the NeRF Uncertainty Deterministic NN model can be visualized using uncertainty visualization tool [NeRFDeltaView Deterministic NN](https://github.com/CTW121/NeRFDeltaView-Deterministic-NN).

Uncertainty visualization provides users with an in-depth understanding of the data for analysis and to perform confident and informed decision-making. The main purpose of our tool is to highlight the significance of interactive visualization in enabling users to explore the estimated uncertainty in synthesized scenes, identify model limitations, and aid in understanding NeRF model uncertainty.