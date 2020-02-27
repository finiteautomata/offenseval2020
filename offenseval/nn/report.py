class EvaluationReport:
    """
    Class that serves as an EvaluationReport of a model
    """

    def __init__(self, acc, pos_f1, neg_f1, loss=None):
        self.macro_f1 = (pos_f1 + neg_f1) / 2
        self.pos_f1 = pos_f1
        self.neg_f1 = neg_f1
        self.loss = loss
        self.acc = acc


    def __repr__(self):
        ret = ""
        if self.loss:
            ret += f'Loss: {self.loss:.3f}'
        ret += f' Acc: {self.acc*100:.2f}%'
        ret += f' Macro F1 {self.macro_f1:.3f} (P {self.pos_f1:.3f} - N {self.neg_f1:.3f})'
        return ret
