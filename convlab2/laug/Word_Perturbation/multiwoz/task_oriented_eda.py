
import random
import string
import re
from functools import lru_cache
from typing import List, Optional, Tuple, Sequence
from collections import defaultdict
from random import shuffle

random.seed(1)

# stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
              'ours', 'ourselves', 'you', 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his',
              'himself', 'she', 'her', 'hers', 'herself',
              'it', 'its', 'itself', 'they', 'them', 'their',
              'theirs', 'themselves', 'what', 'which', 'who',
              'whom', 'this', 'that', 'these', 'those', 'am',
              'is', 'are', 'was', 'were', 'be', 'been', 'being',
              'have', 'has', 'had', 'having', 'do', 'does', 'did',
              'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
              'because', 'as', 'until', 'while', 'of', 'at',
              'by', 'for', 'with', 'about', 'against', 'between',
              'into', 'through', 'during', 'before', 'after',
              'above', 'below', 'to', 'from', 'up', 'down', 'in',
              'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no',
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
              'very', 's', 't', 'can', 'will', 'just', 'don',
              'should', 'now', '']
stop_words = set(stop_words)

ascii_lowercase_and_space = string.ascii_lowercase + ' '



def get_only_chars(line):
    line = line.lower()
    line = re.sub(r"[’']", '', line)
    line = re.sub(r'[\t\n\-]', " ", line)  # replace hyphens with spaces
    line = re.sub(r'[^a-z ]', ' ', line)
    line = re.sub(' +', ' ', line)
    return line.lstrip(' ')


########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################
from nltk.corpus import wordnet


def random_replacement(words, n, excluding_indexes: Optional[Sequence[int]]=None):
    """
    randomly replace n words with synonyms

    Args:
        words: input words
        n: num of replaced words
        excluding_indexes: these words won't be replaced

    Returns:
        new_words (List[str])
        index_map (Dict[int, int]) map an index in words to an index in new_words
    """

    new_words = words.copy()
    indexes = list(range(len(new_words)))
    forbidden = [False for _ in range(len(new_words))]
    if excluding_indexes is not None:
        for i in excluding_indexes:
            forbidden[i] = True

    word2index = defaultdict(list)
    for i, word in enumerate(words):
        if word not in stop_words and not forbidden[i]:
            word2index[word].append(i)
    random_words = list(word2index)
    random.shuffle(random_words)

    num_replaced = 0
    changes = []
    for random_word in random_words:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            synonym_tokens = [token for token in synonym.split() if token.strip()]
            if len(synonym_tokens) == 1:
                for i in word2index[random_word]:
                    new_words[i] = synonym_tokens[0]
                    indexes[i] = None
            else:
                # if synonym has more than 1 words and simply insert synonym, index map will be wrong.
                for i in word2index[random_word]:
                    changes.append((i, synonym_tokens))
            num_replaced += 1
        if num_replaced >= n:  # only replace up to n words
            break

    if changes:
        changes.sort(key=lambda x: x[0])
        offset = 0
        for i, synonym_tokens in changes:
            i += offset
            new_words[i:i+1] = synonym_tokens
            indexes[i:i+1] = [None for _ in range(len(synonym_tokens))]
            offset += len(synonym_tokens) - 1
    return new_words, {v: i for i, v in enumerate(indexes) if v is not None}


def replacement(words, index: int):
    # returns: new_words, start, end, synonym_tokens
    # new_words[start: end+1] == synonym_tokens
    new_words = words.copy()
    word = words[index]
    synonyms = get_synonyms(word)
    if len(synonyms) > 0:
        synonym = random.choice(synonyms)
        synonym_tokens = [token for token in synonym.split() if token.strip()]
        if len(synonym_tokens) == 1:
            new_words[index] = synonym_tokens[0]
            return new_words, index, index, synonym_tokens
        else:
            new_words[index: index+1] = synonym_tokens
            return new_words, index, index + len(synonym_tokens) - 1, synonym_tokens
    else:
        return None


@lru_cache(maxsize=1000)
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join(char for char in synonym if char in ascii_lowercase_and_space).strip()
            if synonym:
                synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p, excluding_indexes: Optional[Sequence[int]]=None):
    """
    remove each word with probability p.

    Args:
        words: input words
        p: delete probability
        excluding_indexes: these words won't be removed.

    Returns:

    """
    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words, {0: 0}

    # randomly delete words with probability p
    new_words = []
    indexes = []
    forbidden = [False for _ in range(len(words))]
    if excluding_indexes is not None:
        for i in excluding_indexes:
            forbidden[i] = True
    for i, word in enumerate(words):
        if forbidden[i]:
            remained = True
        else:
            remained = random.uniform(0, 1) > p
        if remained:
            new_words.append(word)
            indexes.append(i)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]], {rand_int: 0}

    return new_words, {v: i for i, v in enumerate(indexes)}


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n, excluding_indexes: Optional[Sequence[int]]=None):
    """
    randomly swap n pairs of words

    Args:
        words: input words
        n: num of pairs
        excluding_indexes: these words won't be swapped

    Returns:

    """
    new_words = words.copy()
    indexes = list(range(len(words)))
    if excluding_indexes is not None:
        allow_indexes = set(range(len(words))) - set(excluding_indexes)
        allow_indexes = list(allow_indexes)
    else:
        allow_indexes = indexes.copy()

    for _ in range(n):
        new_words = swap_word(new_words, indexes, allow_indexes)
    return new_words, {v: i for i, v in enumerate(indexes)}


def swap_word(new_words, indexes, allow_indexes):
    if len(allow_indexes) <= 1:
        return new_words
    for _ in range(4):
        i = random.choice(allow_indexes)
        j = random.choice(allow_indexes)
        if i != j:
            new_words[i], new_words[j] = new_words[j], new_words[i]
            indexes[i], indexes[j] = indexes[j], indexes[i]
            break
    return new_words


########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n, excluding_indexes: Optional[Sequence[int]]=None):
    """
    randomly insert n words.
    """
    new_words = words.copy()
    indexes = list(range(len(new_words)))
    forbidden = [False for _ in range(len(new_words))]
    if excluding_indexes is not None:
        for i in excluding_indexes:
            forbidden[i] = True

    for _ in range(n):
        add_word(new_words, indexes, forbidden)
    return new_words, {v: i for i, v in enumerate(indexes) if v is not None}


def add_word(new_words, indexes, forbidden):
    if sum(forbidden) == len(new_words):
        return
    synonyms = []
    counter = 0

    while len(synonyms) < 1:
        counter += 1
        if counter >= 15:
            return

        idx = random.randint(0, len(new_words) - 1)
        old_idx = indexes[idx]
        if old_idx is None or forbidden[old_idx]:
            continue
        random_word = new_words[idx]
        synonyms = get_synonyms(random_word)

    random_synonym = synonyms[0]
    for _ in range(5):
        idx = random.randint(0, len(new_words) - 1)
        old_idx = indexes[idx]
        if old_idx is None or not forbidden[old_idx]:
            random_synonym_tokens = [token for token in random_synonym.split() if token.strip()]
            # new_words.insert(idx, random_synonym)
            # indexes.insert(idx, None)
            new_words[idx:idx] = random_synonym_tokens
            indexes[idx:idx] = [None for _ in range(len(random_synonym_tokens))]
            return


########################################################################
# main data augmentation function
########################################################################

def eda(words, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9, excluding_indexes: Optional[Sequence[int]]=None) -> List[Tuple[list, dict]]:
    # sentence = get_only_chars(sentence)
    # words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)

    augmented_sentences: List[Tuple[list, dict]] = []
    num_new_per_technique = int(num_aug / 4) + 1
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    seen = set()
    seen.add(tuple(words))

    # sr
    for _ in range(num_new_per_technique):
        a_words, index_map = random_replacement(words, n_sr, excluding_indexes)
        if tuple(a_words) not in seen:
            seen.add(tuple(a_words))
            augmented_sentences.append((a_words, index_map))

    # ri
    for _ in range(num_new_per_technique):
        a_words, index_map = random_insertion(words, n_ri, excluding_indexes)
        if tuple(a_words) not in seen:
            seen.add(tuple(a_words))
            augmented_sentences.append((a_words, index_map))

    # rs
    for _ in range(num_new_per_technique):
        a_words, index_map = random_swap(words, n_rs, excluding_indexes)
        if tuple(a_words) not in seen:
            seen.add(tuple(a_words))
            augmented_sentences.append((a_words, index_map))

    # rd
    for _ in range(num_new_per_technique):
        a_words, index_map = random_deletion(words, p_rd, excluding_indexes)
        if tuple(a_words) not in seen:
            seen.add(tuple(a_words))
            augmented_sentences.append((a_words, index_map))

    # augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    # trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    return augmented_sentences
