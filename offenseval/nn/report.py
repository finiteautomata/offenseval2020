import torch
from sklearn.metrics import accuracy_score, f1_score

class EvaluationReport:
    """
    Class that serves as an EvaluationReport of a model
    """
    @classmethod
    def from_probas_and_labels(cls, probas, labels, loss=None):

        preds = torch.round(probas).numpy()
        pos_f1 = f1_score(labels, preds)
        neg_f1 = f1_score(1-labels, 1-preds)
        acc = accuracy_score(labels, preds)

        return cls(
            loss=loss, acc=acc, pos_f1=pos_f1, neg_f1=neg_f1,
            probas=probas, labels=labels
        )

    def __init__(self, acc, pos_f1, neg_f1, loss=None, probas=None, labels=None):
        self.macro_f1 = (pos_f1 + neg_f1) / 2
        self.pos_f1 = pos_f1
        self.neg_f1 = neg_f1
        self.loss = loss
        self.acc = acc
        self.probas = probas
        self.labels = labels


    def __repr__(self):
        ret = ""
        if self.loss:
            ret += f'Loss: {self.loss:.3f} '
        ret += f'Acc: {self.acc*100:.2f}%'
        ret += f' Macro F1 {self.macro_f1:.3f} (P {self.pos_f1:.3f} - N {self.neg_f1:.3f})'
        return ret
