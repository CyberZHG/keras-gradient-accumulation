import os
import tempfile
from unittest import TestCase

import numpy as np
from tensorflow.python.keras import optimizers

from keras_gradient_accumulation.backend import keras, TF_KERAS
from keras_gradient_accumulation import GradientAccumulation


class TestGradientAccumulation(TestCase):

    @staticmethod
    def gen_linear_model(optimizer: keras.optimizers.Optimizer) -> keras.models.Model:
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(input_shape=(5,), units=3, use_bias=False, name='Dense'))
        model.compile(optimizer, loss='mse')
        np.random.seed(0xcafe)
        model.get_layer('Dense').set_weights([np.random.standard_normal((5, 3))])
        """
        [[ 0.7562695  -0.7532438  -0.28822958]
         [-1.6990372  -0.09864165 -0.5235034 ]
         [-1.6001531   0.03441733 -0.368053  ]
         [ 1.1673601  -0.69144595 -0.764503  ]
         [ 2.0434828  -0.2743643   0.04834289]]
        """
        return model

    @staticmethod
    def gen_linear_data() -> (np.ndarray, np.ndarray):
        np.random.seed(0xcafe)
        x = np.random.standard_normal((256 * np.random.randint(10, 20), 5))
        w = np.random.standard_normal((5, 3))
        y = np.dot(x, w)
        return x, y

    def _test_accumulation(self, optimizer, **kwargs):
        x, y = self.gen_linear_data()

        model = self.gen_linear_model(optimizer)
        model.fit(x, y, batch_size=128)
        expected = model.get_layer('Dense').get_weights()[0]

        model = self.gen_linear_model(optimizer)
        model.fit(x, y, batch_size=16)
        unwanted = model.get_layer('Dense').get_weights()[0]

        max_diff = np.max(np.abs(unwanted - expected))
        print(max_diff)
        self.assertFalse(np.allclose(unwanted, expected, atol=0.1), (unwanted, expected))

        model = self.gen_linear_model(GradientAccumulation(optimizer, 128, **kwargs))
        if not TF_KERAS:
            model_path = os.path.join(tempfile.gettempdir(), 'test_accumulation_%f.h5' % np.random.random())
            model.save(model_path)
            model = keras.models.load_model(model_path, custom_objects={'GradientAccumulation': GradientAccumulation})
        model.fit(x, y, batch_size=1)
        actual = model.get_layer('Dense').get_weights()[0]

        max_diff = np.max(np.abs(actual - expected))
        print(max_diff)

        self.assertTrue(np.allclose(actual, expected, atol=0.1), (actual, expected))

    def test_sgd(self):
        if TF_KERAS:
            optimizer = optimizers.SGD()
        else:
            optimizer = 'sgd'
        self._test_accumulation(optimizer)

    def test_rmsprop(self):
        if TF_KERAS:
            optimizer = optimizers.RMSprop()
        else:
            optimizer = 'rmsprop'
        self._test_accumulation(optimizer)

    def test_adam(self):
        if TF_KERAS:
            optimizer = optimizers.Adam()
        else:
            optimizer = 'adam'
        self._test_accumulation(optimizer)
