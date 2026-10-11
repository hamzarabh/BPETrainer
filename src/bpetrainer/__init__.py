from datasets import load_dataset
from collections import defaultdict
from functools import lru_cache
import json

class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}

    
    def gen_vocab_freqs(self, text: str):
        vocab_freqs = defaultdict(int)
        for word in text.split():
            wor = ' '.join(["Ġ"]+ list(word))
            vocab_freqs[wor] += 1
        return vocab_freqs

    def get_stats(self, vocab_freqs):
        pair_freqs = defaultdict(int)
        for word , freq in vocab_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) -1):
                pair = (symbols[i] , symbols[i+1])
                pair_freqs[pair] += freq
        return pair_freqs

    def ft_merger(self, pair_freqs, vocab_freq, merges_list):
        c1, c2 = max(dict(pair_freqs), key=pair_freqs.get)
        token = c1 + c2
        merges_list.append([c1,c2])
        new_vocab_freq = defaultdict()
        for word, freq in vocab_freq.items():
            new_word: str = word.replace(f"{c1} {c2}", token)
            new_vocab_freq[new_word] = freq
        return (new_vocab_freq, merges_list)

    def train(self, dataset, size):
        print("\n🚀 Starting BPE training... Please wait!\n")
        vocab_freq = self.gen_vocab_freqs(dataset)
        merges_list = []
        all_tokens: list = []
        for _ in range(size):
            for word in vocab_freq.keys():
                for symbol in word.split():
                    if symbol not in all_tokens:
                        all_tokens.append(symbol)
            pair_freqs = self.get_stats(vocab_freq)
            if not pair_freqs:
                break
            vocab_freq, merges_list = self.ft_merger(pair_freqs, vocab_freq, merges_list)

        vocab_dict = {token:idx for idx, token in enumerate(all_tokens)}
        tokenazer_data = {
            "vocab": vocab_dict,
            "merges":merges_list
        }
        with open('tokenizer.json', 'w', encoding='utf-8') as f:
            json.dump(tokenazer_data, f, ensure_ascii=False, indent=4)
        print("\n✅ BPE training completed successfully!")
    @lru_cache(maxsize=None)
    def encode(self, text: str):
        ids_list = []
        text = text.replace(" ", "Ġ")
        text = list(text)
        with open('tokenizer.json') as f:
            tokenizer_file = json.load(f)
            for a, b in tokenizer_file["merges"]:
                new = f"{a}{b}"
                i = 0
                while i < len(text) - 1:
                    if a == text[i] and b == text[i+1]:
                        text[i] = new
                        del text[i+1]
                    i += 1
            for token in text:
                if token in tokenizer_file["vocab"]:
                    ids_list.append(tokenizer_file["vocab"][token])
                else:
                    continue
            return tuple(ids_list)

    @lru_cache(maxsize=None)
    def decode(self, ids: tuple) -> str:
        with open('tokenizer.json') as f:
            vocab = json.load(f)["vocab"]
            reversed_vocab = dict(zip(vocab.values(), vocab.keys()))
        text_result = ""
        for id in ids:
            text_result += reversed_vocab[id]
        text_result = text_result.replace("Ġ", " ")
        return text_result

def getdata(max_lines: int = 50000) -> str:
    dataset = load_dataset(
        "Salesforce/wikitext", 
        "wikitext-103-raw-v1", 
        split="train"
    )
    lines_written = 0
    with open("data.txt", "w", encoding="utf-8") as f:
        for row in dataset:
            if lines_written >= max_lines:
                break
            text = row["text"]
            if text.strip():
                f.write(text + "\n")
                lines_written += 1

    print(f"Done! Written {lines_written} lines to data.txt.")
    return "data.txt"

