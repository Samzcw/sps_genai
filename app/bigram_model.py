from collections import defaultdict, Counter
import random
import re

class BigramModel:
    def __init__(self, text, frequency_threshold=None):
        self.vocab = []
        self.bigram_probs = defaultdict(dict)
        self.analyze_bigrams(text, frequency_threshold)

    def simple_tokenizer(self, text, frequency_threshold=None):
        if isinstance(text, list):
            text = " ".join(text)

        tokens = re.findall(r"\b\w+\b", text.lower())
        if not frequency_threshold:
            return tokens

        word_counts = Counter(tokens)
        filtered_tokens = [
            token for token in tokens
            if word_counts[token] >= frequency_threshold
        ]

        return filtered_tokens

    def analyze_bigrams(self, text, frequency_threshold=None):
        words = self.simple_tokenizer(text, frequency_threshold)
        bigrams = list(zip(words[:-1], words[1:]))

        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        self.vocab = list(unigram_counts.keys())

        for (word1, word2), count in bigram_counts.items():
            self.bigram_probs[word1][word2] = count / unigram_counts[word1]
    
    def generate_text(self, start_word, length=20):
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current_word)
            if not next_words:
                break

            next_word = random.choices(
                list(next_words.keys()),
                weights=list(next_words.values())
            )[0]
            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)