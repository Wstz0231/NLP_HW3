import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example

# possible keyboard typos for vowels
vowel_key = {'a': ['q', 's', 'z'],
             'e': ['w', 'd', 'r'],
             'i': ['u', 'k', 'o'],
             'o': ['i', 'l', 'p'],
             'u': ['y', 'j', 'i']}


# introduce a vowel typo in the word
def vowel_typo(word):
    vowels = [i for i, letter in enumerate(word) if letter.lower() in vowel_key]
    if vowels:
        idx = random.choice(vowels)
        letter = word[idx].lower()
        typo = random.choice(vowel_key[letter])
        word = word[:idx] + typo + word[idx+1:]
    return word


# for word with len > 3, introduce missing letter
def missing_letter(word):
    if len(word) > 3:
        idx = random.randint(0, len(word)-1)
        word = word[:idx] + word[idx+1:]
    return word

# provide a synonym for the word
def get_syn(word):
    synset = wordnet.synsets(word)
    if synset:
        syns = [lemma.name() for lemma in synset[0].lemmas()]
        if syns:
            return random.choice(syns).replace("_", " ")
    return None


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]
    words = word_tokenize(text)
    trans_words = []
    for word in words:
        out = word
        if random.random() < 0.2:
            choice = random.choice(["syn_replace", "vowel", "missing"])
            # synonym replacement
            if choice == "syn_replace":
                syn = get_syn(word)
                if syn:
                    out = syn
            # vowel typo
            elif choice == "vowel":
                out = vowel_typo(word)
            # missing letter
            else:
                out = missing_letter(word)
        trans_words.append(out)
    example["text"] = TreebankWordDetokenizer().detokenize(trans_words)
    ##### YOUR CODE ENDS HERE ######

    return example
