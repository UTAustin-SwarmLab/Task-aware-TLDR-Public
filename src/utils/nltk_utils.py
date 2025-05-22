from nltk import pos_tag
from nltk.tokenize import word_tokenize

### Only run once to download the required files
# import nltk
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('punkt_tab')


def IsNoun(sentence: str) -> list[tuple[str, bool]]:
    """ "
    This function takes a sentence as input and returns a list of tuples where each tuple contains a word and a boolean
    value indicating whether the word is a noun or not.

    Input : sentence : str : A sentence
    Output : list[tuple[str, bool]] : A list of tuples where each tuple contains a word and a boolean value indicating
    whether the word is a noun or not.
    """
    # NN noun, singular ‘- desk’
    # NNS noun plural – ‘desks’
    # NNP proper noun, singular – ‘Harrison’
    # NNPS proper noun, plural – ‘Americans’
    tokens = word_tokenize(sentence)
    tagged_tokens = pos_tag(tokens)
    noun_TF = []
    for word, tag in tagged_tokens:
        print(word, tag)
        noun_TF.append((word, tag == "NN" or tag == "NNS" or tag == "NNP" or tag == "NNPS"))
    return noun_TF


if __name__ == "__main__":
    text = "Your time is limited, so don't waste it living someone else's life and crying."
    print(IsNoun(text))
