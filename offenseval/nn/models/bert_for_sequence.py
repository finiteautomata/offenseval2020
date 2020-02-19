import torch.nn as nn
from transformers import BertPreTrainedModel

class BertModelWithAdapter(nn.Module):
    def __init__(self, bert, dropout=0.1, num_labels=1):
        """
        Arguments:
        ---------

        bert: BertModel
        """
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            bert.config.hidden_size,
            num_labels
        )


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        adapter=None,
    ):
        """
        Adapter: a function
            Function to be applied between BERT and the linear layer
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        if adapter:
            pooled_output = adapter(pooled_output)

        out = self.classifier(pooled_output)

        return out
