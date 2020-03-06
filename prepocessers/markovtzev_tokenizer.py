import Stemmer
from enum import Enum
import functools
import re


class TokenStyle(Enum):
    """Metadata that should allow to reconstruct initial identifier from a list of tokens."""
    DELIMITER = 1
    TOKEN_UPPER = 2
    TOKEN_LOWER = 3
    TOKEN_CAPITALIZED = 4


class TokenParser:

    NAME_BREAKUP_RE = re.compile(r"[^a-zA-Z]+")
    NAME_BREAKUP_KEEP_DELIMITERS_RE = re.compile(r"([^a-zA-Z]+)")

    def __init__(self, stem_threshold=6, max_token_length=256, min_split_length=3, save_token_style=False,
                 single_shot=False, attach_upper=True):
        self.stemmer = Stemmer.Stemmer("english")
        self.stem_threshold = stem_threshold
        self.max_token_length = max_token_length
        self.min_split_length = min_split_length
        self.save_token_style = save_token_style
        self.single_shot = single_shot
        self.attach_upper = attach_upper

    def stem(self, word):
        if len(word) <= self.stem_threshold:
            return word
        return self.stemmer.stemWord(word)

    def split(self, token: str) -> [str]:
        yield from self._split(token)

    def _split(self, token):
        token = token.strip()[:self.max_token_length]

        def meta_decorator(func):
            if self.save_token_style:
                @functools.wraps(func)
                def decorated_func(name):
                    if name.isupper():
                        meta = TokenStyle.TOKEN_UPPER
                    elif name.islower():
                        meta = TokenStyle.TOKEN_LOWER
                    else:
                        meta = TokenStyle.TOKEN_CAPITALIZED
                    for res in func(name):
                        yield res, meta
                return decorated_func
            else:
                return func

        @meta_decorator
        def ret(name):
            r = name.lower()
            if len(name) >= self.min_split_length:
                ret.last_subtoken = r
                yield r
                if ret.prev_p and not self.single_shot:
                    yield ret.prev_p + r
                    ret.prev_p = ""
            elif not self.single_shot:
                ret.prev_p = r
                yield ret.last_subtoken + r
                ret.last_subtoken = ""
        ret.prev_p = ""
        ret.last_subtoken = ""

        if self.save_token_style:
            regexp_splitter = self.NAME_BREAKUP_KEEP_DELIMITERS_RE
        else:
            regexp_splitter = self.NAME_BREAKUP_RE

        for part in regexp_splitter.split(token):
            if not part:
                continue
            if self.save_token_style and not part.isalpha():
                yield part, TokenStyle.DELIMITER
                continue
            assert part.isalpha()
            start = 0
            for i in range(1, len(part)):
                this = part[i]
                prev = part[i - 1]
                if prev.islower() and this.isupper():
                    yield from ret(part[start:i])
                    start = i
                elif prev.isupper() and this.islower():
                    if self.attach_upper and i > 1 and part[i - 2].isupper():
                        new_start = i - 1
                    else:
                        new_start = i
                    if i - 1 > start:
                        yield from ret(part[start:new_start])
                        start = new_start
            last = part[start:]
            if last:
                yield from ret(last)


if __name__ == "__main__":
    token_parser = TokenParser()
    print(token_parser.stem('attached'))
    print(list(token_parser.split('getstringfromfile')))