from .backend import keras, optimizers
from .backend import backend as K

__all__ = ['GradientAccumulation']


class GradientAccumulation(keras.optimizers.Optimizer):
    """Optimizer wrapper for gradient accumulation.

    # Arguments
        optimizer: Original optimizer.
        accumulation_steps: int > 0. Update gradient in every accumulation steps.
        momentum_names: A collection of strings. Names of momentum terms.
    """

    def __init__(self, optimizer, accumulation_steps, momentum_names=None, **kwargs):
        super(GradientAccumulation, self).__init__(**kwargs)
        self.optimizer = optimizers.get(optimizer)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.accumulation_steps = K.variable(accumulation_steps, dtype='int64', name='accumulation_steps')
        if momentum_names is None:
            momentum_names = ['momentum', 'rho', 'beta_1', 'beta_2']
        self.momentum_names = momentum_names
        self._lr = self.optimizer.lr

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, lr):
        self._lr = lr

    def get_updates(self, loss, params):
        # Create accumulated gradients
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        mod = self.iterations % self.accumulation_steps
        update_cond = K.equal(mod, 0)
        acc_grads = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        for grad, acc_grad in zip(grads, acc_grads):
            self.updates.append(K.update(
                acc_grad,
                K.switch(update_cond, grad, acc_grad + (grad - acc_grad) / K.cast(mod, K.floatx())),
            ))
        self.optimizer.get_gradients = lambda _loss, _params: \
            [K.switch(update_cond, grad / K.cast(self.accumulation_steps, K.floatx()), K.zeros_like(grad))
             for grad in acc_grads]

        # Use fake iterations
        fake_iterations = self.iterations // self.accumulation_steps
        self.optimizer.iterations = fake_iterations
        original_update_add = getattr(K, 'update_add')
        setattr(
            K,
            'update_add',
            lambda x, increment: original_update_add(x, increment) if x is not fake_iterations else None,
        )

        # Use fake learning rate
        self.optimizer.lr = K.switch(update_cond, self.lr, 0.0)

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
        self.optimizer.lr = self.lr

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
