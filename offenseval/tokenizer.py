import html
import re

class Tokenizer:
    """
    Tokenizer for tweets based on BERT Tokenizer + NLTK's Tokenizer
    """
    def __init__(self, bert_tokenizer, html_unescape=True, max_len=128):
        """
        Arguments:
        ----------
        html_unescape: Boolean (default False)
            Use or not `html.unescape` on text before tokenizing
        """
        self._html_unescape = html_unescape
        self.bert_tokenizer = bert_tokenizer
        self.max_len = max_len

        self._patterns = {
            "<hour>": re.compile(r"\d{1,2}\:\d{2}"),
            "<year>": re.compile(r"(1(7|8|9)|2(0|1))\d\d"),
            "<num>": re.compile(r"\d+(\.)?\d*"),
            # @stephenhay's from https://mathiasbynens.be/demo/url-regex
            "<url>": re.compile(r"(https?|ftp)://[^\s/$.?#].[^\s]*"),
        }

    def replace_patterns(self, text):
        for repl, pattern in self._patterns.items():
            text = pattern.sub(repl, text)
        return text

    def tokenize(self, text):
        if self._html_unescape:
            text = html.unescape(text)

        text = self.replace_patterns(text)
        return self.bert_tokenizer.tokenize(text)[:self.max_len]

    def convert_tokens_to_ids(self, *args, **kwargs):
        return self.bert_tokenizer.convert_tokens_to_ids(*args, **kwargs)
