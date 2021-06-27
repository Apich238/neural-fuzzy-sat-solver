import torch

from torch.nn import Module, Linear, LSTMCell
import torch.nn.functional as F


class FuzzyEvaluation(Module):
    def __init__(self):
        super().__init__()

    def forward(self, mxs, cons, negs, truth_degrees, ops, rnn_steps=16):
        a = torch.zeros([mxs.shape[0], ops], dtype=torch.float32, device=mxs.device)
        for i in range(rnn_steps):
            g = mxs * torch.cat([a, truth_degrees], -1).reshape([mxs.shape[0], 1, -1])
            h = self.eval_ops(g, mxs, cons)
            a = (1 - h) * negs + h * (1 - negs)
        return a[:, 0]

    def eval_ops(self, g, m, con):
        pass


class GodelEvaluation(FuzzyEvaluation):

    def __init__(self):
        super().__init__()

    def eval_ops(self, g, m, con):
        conjunctions = torch.min(torch.max(1 - m, g), dim=2).values
        disjunctions = torch.max(g, dim=2).values
        r = con * conjunctions + (1 - con) * disjunctions
        return r


class ProbabilisticEvaluation(FuzzyEvaluation):

    def __init__(self):
        super().__init__()

    def eval_ops(self, g, m, con):
        conjunctions = torch.prod(torch.max(1 - m, g), dim=2)
        disjunctions = 1 - torch.prod(1 - g, dim=2)
        r = con * conjunctions + (1 - con) * disjunctions
        return r


class LukasieviczEvaluation(FuzzyEvaluation):

    def __init__(self):
        super().__init__()

    def eval_ops(self, g, m, con):
        conjunctions = torch.clamp_min(torch.sum(torch.max(1 - m, g), dim=2) - g.shape[2] + 1, 0)
        disjunctions = torch.clamp_max(torch.sum(g, dim=2), 1)
        r = con * conjunctions + (1 - con) * disjunctions
        return r


class SimpleTreeSAT(Module):
    def __init__(self, emb_dim, classifier_type):
        """

        :param emb_dim:
        :param classifier_type: тип классификатора:
            1 - по состоянию верхней вершины,
            2 - по состоянию всех вершин,
            3 - по состоянию только вершин-переменных,
            5 - вычисление по логике Гёделя (Заде, минимаксной),
            6 - вычисление по вероятностной логике,
            7 - вычисление по логике Лукашевича
        """
        super().__init__()
        self.classifier_type = classifier_type
        self.emb_dim = emb_dim
        self.start_embeddings = torch.nn.Parameter(torch.FloatTensor(self.emb_dim))
        torch.nn.init.normal_(self.start_embeddings, 0, 1)

        self.con_embeddings = torch.nn.Parameter(torch.FloatTensor(self.emb_dim))
        torch.nn.init.normal_(self.con_embeddings, 0, 1)

        self.neg_embeddings = torch.nn.Parameter(torch.FloatTensor(self.emb_dim))
        torch.nn.init.normal_(self.neg_embeddings, 0, 1)

        self.var_embeddings = torch.nn.Parameter(torch.FloatTensor(self.emb_dim))
        torch.nn.init.normal_(self.var_embeddings, 0, 1)

        self.msg1_1_f = Linear(2 * self.emb_dim, 2 * self.emb_dim, True)
        self.msg1_2_f = Linear(2 * self.emb_dim, self.emb_dim, True)

        self.msg2_1_f = Linear(2 * self.emb_dim, 2 * self.emb_dim, True)
        self.msg2_2_f = Linear(2 * self.emb_dim, self.emb_dim, True)

        self.update_f = LSTMCell(self.emb_dim, self.emb_dim, self.emb_dim)

        self.clf_1 = Linear(self.emb_dim, self.emb_dim, True)
        self.clf_2 = Linear(self.emb_dim, 1, False)

        if self.classifier_type in [1, 2, 3, 4]:
            pass
        elif self.classifier_type == 5:
            self.cl = GodelEvaluation()
        elif self.classifier_type == 6:
            self.cl = ProbabilisticEvaluation()
        elif self.classifier_type == 7:
            self.cl = LukasieviczEvaluation()

    def forward(self, mxs, cons, negs, rnn_steps=16):
        mxs = mxs.to(torch.float32)
        bs = mxs.shape[0]
        ops = mxs.shape[1]
        vars = mxs.shape[2] - mxs.shape[1]

        mxs2 = F.pad(mxs, [0, 0, 0, vars, 0, 0])
        mxs2T = mxs2.transpose(1, 2)

        v_states = torch.cat(bs * [torch.cat((ops + vars) * [self.start_embeddings.unsqueeze(0)]).unsqueeze(0)])

        v_types = torch.zeros([bs, ops + vars, self.emb_dim], dtype=torch.float32, device=v_states.device)
        v_types[:, :ops] = cons.unsqueeze(2) * self.con_embeddings + negs.unsqueeze(2) * self.neg_embeddings
        v_types[:, ops:] = self.var_embeddings.unsqueeze(0)

        upd_hidden_and_cell = None

        for i in range(rnn_steps):
            x = torch.cat([v_states, v_types], 2)

            msg1 = torch.relu(self.msg1_1_f(x))
            msg1 = self.msg1_2_f(msg1)

            msg2 = torch.relu(self.msg2_1_f(x))
            msg2 = self.msg2_2_f(msg2)

            sum1 = mxs2.matmul(msg1)
            sum2 = mxs2T.matmul(msg2)

            upd_hidden_and_cell = self.update_f((sum1 + sum2).reshape([-1, self.emb_dim]), upd_hidden_and_cell)
            v_states = upd_hidden_and_cell[0].reshape([bs, ops + vars, self.emb_dim])

        if self.classifier_type == 1: # make a prediction based only on root node
            x = torch.relu(self.clf_1(v_states[:, 0]))
            x = torch.sigmoid(self.clf_2(x))[:, 0]
            res = x
        elif self.classifier_type == 2: # make a prediction based mean of all nodes
            x = torch.relu(self.clf_1(v_states))
            x = torch.sigmoid(self.clf_2(x))
            res = torch.mean(x, [1, 2], False)
        elif self.classifier_type == 3: # make a prediction based mean of all variables nodes
            x = torch.relu(self.clf_1(v_states[:, ops:]))
            x = torch.sigmoid(self.clf_2(x))
            res = torch.mean(x, [1, 2], False)
        else: # predict truth degree and evaluate formula on a fuzzy logic algebra
            x = torch.relu(self.clf_1(v_states[:, ops:]))
            x = torch.sigmoid(self.clf_2(x))[:, :, 0]
            tvalues = x
            res = self.cl(mxs, cons, negs, tvalues, ops, ops)

        return res
