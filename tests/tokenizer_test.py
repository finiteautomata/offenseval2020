import pytest
from transformers import BertTokenizer
from offenseval.nn import Tokenizer

@pytest.fixture
def bert_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

@pytest.fixture
def tokenizer(bert_tokenizer):
    return Tokenizer(bert_tokenizer)

def test_tokenizes_basic_sentence(tokenizer):
    assert tokenizer.tokenize("hello world") == ["hello", "world"]

def test_converts_numbers_to_special_token(tokenizer):
    assert tokenizer.tokenize("hello 111 world") == ["hello", "<", "num", ">", "world"]


def test_splits_two_words(tokenizer):

    tokens = tokenizer.tokenize("HELLO/WORLD")

    assert tokens == ["hello", "/", "world"]

def test_converts_urls(tokenizer):
    tokens = tokenizer.tokenize("HELLO WORLD http://t.co/sarasa")
    assert tokens == ["hello", "world", "<", "url", ">"]
