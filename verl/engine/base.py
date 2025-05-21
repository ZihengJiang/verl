class Engine(object):
    def __init__(self, config):
        pass

    def init_model_and_optimizer(self):
        raise NotImplementedError

    def forward_backward_step(self, batch):
        raise NotImplementedError

    def optimizer_zero_grad(self):
        raise NotImplementedError

    def optimizer_step(self):
        raise NotImplementedError

    def lr_scheduler_step(self):
        raise NotImplementedError
    
    def set_loss_fn(self, loss_fn):
        raise NotImplementedError