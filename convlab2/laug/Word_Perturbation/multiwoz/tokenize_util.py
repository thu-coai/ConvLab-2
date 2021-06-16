from .types import TokenListType, SentenceType


def tokenize(sentence: str) -> TokenListType:
    return [token for token in sentence.split() if token.strip()]

def convert_sentence_to_tokens(sentence: SentenceType) -> TokenListType:
    if isinstance(sentence, str):
        return tokenize(sentence)
    else:
        assert isinstance(sentence, list)
        return sentence

def convert_tokens_to_string(tokens: TokenListType) -> str:
    return ' '.join(tokens)
