import pytest
import torch
from unittest.mock import MagicMock
from offenseval.nn.models import BertSeqModel


@pytest.fixture
def bert_model():
    bert_model = MagicMock()

    bert_model.config.hidden_size = 768

    def bert_return(inp, **kwargs):
        pooled_output = torch.randn(inp.shape[0], 768)
        hidden = torch.randn(inp.shape[0], inp.shape[1], 768)
        return hidden, pooled_output

    bert_model.side_effect = bert_return
    return bert_model

@pytest.fixture
def model(bert_model):
    return BertSeqModel(bert_model)


def test_it_can_be_created(bert_model):
    model = BertSeqModel(bert_model)

def test_it_returns_something(model):
    # (batch, seqlen)
    input = torch.LongTensor(32, 60)

    out = model(input)
    print(out.shape)
    assert out.shape == torch.Size([32, 1])

def test_it_applies_adapter_if_given(model):
    input = torch.LongTensor(32, 60)

    """
    Create adapter that replaces the out of BERT by something else
    """
    adapter = MagicMock()
    ret = torch.ones(10, 768)
    adapter.side_effect = lambda x: ret

    out = model(input, adapter=adapter)
    assert all(torch.eq(out, model.classifier(ret)))
