import torch
import torch.nn as nn
import torch.nn.functional as F

def gumbel_softmax(logits, temperature=1.0, hard=False):
    gumbels = -torch.empty_like(logits).exponential_().log()
    x = (logits + gumbels) / temperature
    y_soft = F.softmax(x, dim=-1)
    if hard:
        _, index = y_soft.max(-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

class EncoderModel(nn.Module):
    """ Model for learning the backward kinematic inseparability """

    NAME = "backwardmodel"

    def __init__(self, dO, nA, nZ, hidden_dim):
        super().__init__()

        self.nZ = nZ

        self.obs_encoder = nn.Sequential(
            nn.Linear(dO, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, nZ),
        )

        # action embedding
        self.action_emb = nn.Embedding(nA, nA)
        self.action_emb.weight.data.copy_(torch.eye(nA).float())
        self.action_emb.weight.requires_grad = False

        self.abstract_state_emb = nZ
        self.obs_emb_dim = dO
        self.act_emb_dim = nA

        # Model head
        self.classifier = nn.Sequential(
            nn.Linear(nZ + nA + nZ, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def __gen_logits__(self, prev_observations, actions, curr_observations, discretized, type="logsoftmax"):
        prev_encoding = self.obs_encoder(prev_observations)
        curr_encoding = self.obs_encoder(curr_observations)
        action_x = self.action_emb(actions).squeeze()

        prev_z = gumbel_softmax(prev_encoding, hard=discretized)
        curr_z = gumbel_softmax(curr_encoding, hard=discretized)

        x = torch.cat([prev_z, action_x, curr_z], dim=1)
        logits = self.classifier(x)

        if type == "logsoftmax":
            result = F.log_softmax(logits, dim=1)
        elif type == "softmax":
            result = F.softmax(logits, dim=1)
        else:
            raise AssertionError("Unhandled type ", type)

        return result, {}

    def gen_log_prob(self, prev_observations, actions, observations, discretized):
        return self.__gen_logits__(prev_observations, actions, observations, discretized, type="logsoftmax")

    def gen_prob(self, prev_observations, actions, observations, discretized):
        return self.__gen_logits__(prev_observations, actions, observations, discretized, type="softmax")

    def encode_observations(self, observations):
        if self.config["feature_type"] == 'feature':
            assert len(observations.size()) == 1
            observations = observations.view(1, -1)
        else:
            raise NotImplementedError()

        log_prob = F.log_softmax(self.obs_encoder(observations), dim=1)

        argmax_indices = log_prob.max(1)[1]

        return int(argmax_indices[0])

    @staticmethod
    def _freeze_param(parameters):
        for param in parameters:
            param.requires_grad = False

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
