from torch.utils.data import Dataset
import torch
from tokenizers import Tokenizer
import prep_data

class Gpt2ShakespeareDatset(Dataset):
    def __init__(self):
        self.tokenizer: Tokenizer = Tokenizer.from_file("tokenizer.json")
        self.data = prep_data.load_data()
        self.tokens = []
        self.token_ids = []
        self.data_tokens = [self.tokenizer.encode(x) for x in self.data]

        for i in range(len(self.data_tokens)):
            tokens = self.data_tokens[i].tokens
            ids = self.data_tokens[i].ids

            # Append BOS and EOS tokens at the start/end of every play
            tokens.insert(0, "<|bos|>")
            ids.insert(0, 1)
            tokens.append("<|eos|>")
            ids.append(2)
            self.tokens.append(tokens)
            self.token_ids.append(ids)

        # Split tensors into batches of 512 tokens
        tensors = torch.tensor([id_entries for id in self.token_ids for id_entries in id])
        window_size = 512 + 1
        B = tensors.numel() // window_size
        trimmed_tensors = tensors[:B * window_size]
        trimmed_tensors = trimmed_tensors.view(B, window_size)
        # Inputs are all but last token in each window; targets are next tokens
        self.x = trimmed_tensors[:, :-1]
        self.y = trimmed_tensors[:, 1:]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
