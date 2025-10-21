from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import prep_data

tok = Tokenizer(BPE(unk_token="<|unk|>"))
tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
tok.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=8000,
    special_tokens=["<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>", "<|sep|>"],
)
prepped_data = prep_data.load_data()

def lines():
    for x in prepped_data:
        yield x

tok.train_from_iterator(lines(), trainer=trainer)
tok.save("tokenizer.json")
