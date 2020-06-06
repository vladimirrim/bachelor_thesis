class Experiment:
    def __init__(self, **kwargs):
        self.defaults()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def defaults(self):
        self.MODELDIR = 'resource/logs/'
        self.HIDDEN = 128
        self.TEST = False
        self.LOAD = False
        self.BEST = False
        self.HORIZON = 1000
        self.EPOCHS = 1
        self.EPOCHS_PPO = 3
        self.LM_LAMBDA = 0.9
        self.NPOP = 5
        self.NENT = 5
        self.ENTROPY = 0.05
        self.ENTROPY_ANNEALING = 0.998
        self.MIN_ENTROPY = 0.00176
        self.DEVICE_OPTIMIZER = 'cpu'
        self.PPO_MINIBATCH = 40
        self.LR = 0.001
        self.LSTM = True
        self.LSTM_PERIOD = 50
        self.LM_MODE = 'sum'  # sum, min
