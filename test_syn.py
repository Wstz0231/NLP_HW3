from utils import get_syn
import nltk

# Define a test sentence
test_sentence = "The happy dog runs fast in the big park."

# Tokenize the sentence into words
words = test_sentence.split()

# Apply the get_syn function to each word and print results
print("Original Sentence: ", test_sentence)
print("Synonym Replacements:")

for word in words:
    synonym = get_syn(word)
    if synonym:
        print(f"{word} → {synonym}")
    else:
        print(f"{word} → No synonym found")