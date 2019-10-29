# Keras Gradient Accumulation

[![Travis](https://travis-ci.org/CyberZHG/keras-gradient-accumulation.svg)](https://travis-ci.org/CyberZHG/keras-gradient-accumulation)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-gradient-accumulation/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-gradient-accumulation)
[![Version](https://img.shields.io/pypi/v/keras-gradient-accumulation.svg)](https://pypi.org/project/keras-gradient-accumulation/)
![Downloads](https://img.shields.io/pypi/dm/keras-gradient-accumulation.svg)
![License](https://img.shields.io/pypi/l/keras-gradient-accumulation.svg)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)

\[[中文](https://github.com/CyberZHG/keras-gradient-accumulation/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-gradient-accumulation/blob/master/README.md)\]

## Install

```bash
pip install keras-gradient-accumulation
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
