from torch.nn.utils import clip_grad_value_

from forge.trinity.pantheon import Model, Pantheon
from forge.ethyr.torch.optim import backwardAgent
from forge.ethyr.torch.forward import ForwardLm


class ModelLm(Model):
    def __init__(self, config, args):
        super(ModelLm, self).__init__(config, args)

    def backward(self, batch):
        self.lmOpt.zero_grad()
        backwardAgent(batch, device=self.config.device, n_quant=self.config.N_QUANT_LM)
        clip_grad_value_(self.lm.parameters(), 10)
        self.lmOpt.step()

    def checkpoint(self, reward=None):
        if self.config.TEST:
            return
        self.saver.checkpointLawmaker(self.lm, self.lmOpt)


class PantheonLm(Pantheon):
    def __init__(self, config, args):
        super(PantheonLm, self).__init__(config, args)
        self.net = ModelLm(config, args)
        self.forward = ForwardLm(config)

    def step(self, sample, weights):
        if self.config.NOISE:
            self.net.reset_noise()
        batch, priorities = self.forward.forward(sample, weights, self.net.anns, self.net.lm, device=self.config.device)
        self.net.backward(batch)

        self.tick += 1
        if not self.config.TEST:
            if (self.tick + 1) % 256 == 0:
                self.net.checkpoint()

        return self.net.sendLm(), priorities
