import random
import re
from collections import defaultdict

class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel.
        """
        # store counts as dict: context -> {next_word: count}
        # context is a tuple of two words (w1, w2)
        self.counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()
        self.trained = False

    def _clean_and_tokenize_sentence(self, sentence):
        # lowercase and keep only alphanumerics and whitespace
        s = sentence.lower()
        s = re.sub(r'[^a-z0-9\s]', ' ', s)
        tokens = s.split()
        return tokens

    def fit(self, text):
        """
        Trains the trigram model on the given text.

        Args:
            text (str): The text to train the model on.
        """
        # handle empty input
        if not text or not text.strip():
            self.trained = False
            return

        # split into sentences by ., ?, !
        sentences = re.split(r'[.!?]+', text)
        for sent in sentences:
            tokens = self._clean_and_tokenize_sentence(sent)
            if not tokens:
                continue
            # update vocab
            for t in tokens:
                self.vocab.add(t)
            # pad with two start tokens and one end token
            padded = ['<s>', '<s>'] + tokens + ['</s>']
            for i in range(len(padded) - 2):
                context = (padded[i], padded[i+1])
                next_word = padded[i+2]
                self.counts[context][next_word] += 1
                if next_word != '</s>':
                    self.vocab.add(next_word)

        self.trained = True

    def generate(self, max_length=50):
        """
        Generates new text using the trained trigram model.

        Args:
            max_length (int): The maximum length of the generated text.

        Returns:
            str: The generated text.
        """
        if not self.trained:
            return ""

        context = ('<s>', '<s>')
        generated = []
        for _ in range(max_length):
            choices_dict = self.counts.get(context)
            if not choices_dict:
                break
            words = list(choices_dict.keys())
            weights = [choices_dict[w] for w in words]
            next_word = random.choices(words, weights=weights, k=1)[0]
            if next_word == '</s>':
                break
            generated.append(next_word)
            context = (context[1], next_word)

        return " ".join(generated)
