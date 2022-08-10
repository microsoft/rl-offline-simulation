import torch
import torch.nn as nn
import torch.nn.functional as F


class ObservationEncoder():
    def __init__(self, model=None):
        self.model = model

    def train(self):
        pass

    def encode(self, observation):
        pass


class HomerEncoder(ObservationEncoder):
    def __init__(self, model=None, model_path=None):
        if model and model_path:
            model.load(model_path)

        super().__init__(model)

    def train(self):
        self.model.train()

    def encode(self, observation):
        assert len(observation.size()) == 1
        observation = observation.view(1, -1)
        log_prob = F.log_softmax(self.model.obs_encoder(observation), dim=1)
        argmax_indices = log_prob.max(1)[1]
        return int(argmax_indices[0])
