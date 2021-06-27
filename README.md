# Keras Gradient Accumulation

[![Travis](https://travis-ci.org/CyberZHG/keras-gradient-accumulation.svg)](https://travis-ci.org/CyberZHG/keras-gradient-accumulation)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-gradient-accumulation/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-gradient-accumulation)

**This repo is outdated and will no longer be maintained.**

## Install

```bash
pip install git+https://github.com/cyberzhg/keras-gradient-accumulation.git
```

## Usage

### Wrapper

```python
from keras_gradient_accumulation import GradientAccumulation

optimizer = GradientAccumulation('adam', accumulation_steps=8)
```

### Adam

```python
from keras_gradient_accumulation import AdamAccumulated

optimizer = AdamAccumulated(accumulation_steps=8)
```

## Known Issues

* Not available for batch normalization
* Not compatible with `OptimizerV2`
