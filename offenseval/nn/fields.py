from torchtext import data
from .tokenizer import Tokenizer

def create_bert_fields(bert_tokenizer):
    """
    Create fields for torch training for BERT

    Arguments:
    ----------


    bert_tokenizer: BertTokenizer
        Instance of tokenizer

    Returns:
        (ID, SUBTASK_A, TEXT): triple of data.Fields
    """
    init_token_idx = bert_tokenizer.cls_token_id
    eos_token_idx = bert_tokenizer.sep_token_id
    pad_token_idx = bert_tokenizer.pad_token_id
    unk_token_idx = bert_tokenizer.unk_token_id

    # Trying to cut this down to check if this improves memory usage

    tokenizer = Tokenizer(bert_tokenizer)

    ID = data.Field(sequential=False, use_vocab=False)
    # All these arguments are because these are really floats
    # See https://github.com/pytorch/text/issues/78#issuecomment-541203609
    SUBTASK_A = data.LabelField()

    TEXT = data.Field(
        tokenize=tokenizer.tokenize,
        include_lengths = True,
        use_vocab=False,
        batch_first = True,
        preprocessing = tokenizer.convert_tokens_to_ids,
        init_token = init_token_idx,
        eos_token = eos_token_idx,
        pad_token = pad_token_idx,
        unk_token = unk_token_idx
    )

    return ID, SUBTASK_A, TEXT
