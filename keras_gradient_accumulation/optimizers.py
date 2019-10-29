import tensorflow as tf

from .backend import keras, optimizers, TF_KERAS
from .backend import backend as K

__all__ = ['GradientAccumulation']


def identity(x):
    return x


symbolic = identity
if hasattr(K, 'symbolic'):
    symbolic = K.symbolic


class GradientAccumulation(keras.optimizers.Optimizer):
    """Optimizer wrapper for gradient accumulation.

    # Arguments
        optimizer: Original optimizer.
        accumulation_steps: int > 0. Update gradient in every accumulation steps.
        momentum_names: A collection of strings. Names of momentum terms.
    """

    def __init__(self,
                 optimizer,
                 accumulation_steps,
                 momentum_names=None,
                 **kwargs):
        super(GradientAccumulation, self).__init__(**kwargs)
        self.optimizer = optimizers.get(optimizer)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.accumulation_steps = K.variable(accumulation_steps, dtype='int64', name='accumulation_steps')
        if momentum_names is None:
            momentum_names = ['momentum', 'rho', 'beta_1', 'beta_2']
        self.momentum_names = momentum_names
        self._lr = self.optimizer.learning_rate

    @property
    def learning_rate(self):
        return self.optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.optimizer.learning_rate = learning_rate

    @symbolic
    def get_updates(self, loss, params):
        # Create accumulated gradients
        grads = self.get_gradients(loss, params)
        self.updates = []
        with tf.control_dependencies([self.iterations.assign_add(1)]):
            update_cond = K.equal(self.iterations % self.accumulation_steps, 0)
            sub_step = (self.iterations - 1) % self.accumulation_steps + 1
            fake_iterations = (self.iterations - 1) // self.accumulation_steps + 1
        acc_grads = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        for grad, acc_grad in zip(grads, acc_grads):
            ave_grad = grad / K.cast(self.accumulation_steps, K.floatx())
            self.updates.append(K.update(
                acc_grad,
                K.switch(
                    K.equal(sub_step, 1),
                    ave_grad,
                    acc_grad + (ave_grad - acc_grad) / K.cast(sub_step, K.floatx())
                ),
            ))
        self.optimizer.get_gradients = lambda _loss, _params: \
            [K.switch(update_cond, grad, K.zeros_like(grad))
             for grad in acc_grads]

        # Use fake iterations
        original_iterations = self.optimizer.iterations
        if TF_KERAS:
            from tensorflow.python import state_ops
            original_assign_add = getattr(state_ops, 'assign_add')
            setattr(
                state_ops,
                'assign_add',
                lambda ref, val: original_assign_add(ref, val) if ref is not fake_iterations
                else original_assign_add(original_iterations, val)
            )
        else:
            original_update_add = getattr(K, 'update_add')
            setattr(
                K,
                'update_add',
                lambda x, increment: original_update_add(x, increment) if x is not fake_iterations else None,
            )
        self.optimizer.iterations = fake_iterations

        # Use fake learning rate
        self.optimizer.learning_rate = K.switch(update_cond, self.lr, 0.0)

        # Freeze momentum
        momentum = {}
        for name in self.momentum_names:
            if hasattr(self.optimizer, name):
                momentum[name] = getattr(self.optimizer, name)
                setattr(self.optimizer, name, K.switch(update_cond, momentum[name], (1.0 - K.epsilon())))

        for update in self.optimizer.get_updates(loss, params):
            if update is not None:
                self.updates.append(update)

        # Restore variables
        for name, value in momentum.items():
            setattr(self.optimizer, name, value)
        self.optimizer.learning_rate = self._lr
        self.optimizer.iterations = original_iterations
        if TF_KERAS:
            from tensorflow.python import state_ops
            setattr(state_ops, 'assign_add', original_assign_add)
        else:
            setattr(K, 'update_add', original_update_add)

        return self.updates

    def get_config(self):
        config = {
            'optimizer': optimizers.serialize(self.optimizer),
            'accumulation_steps': int(K.get_value(self.accumulation_steps)),
            'momentum_names': self.momentum_names,
        }
        base_config = super(GradientAccumulation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        optimizer = optimizers.deserialize(config.pop('optimizer'))
        return cls(optimizer, **config)
