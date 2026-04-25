from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# buat tokenizer BPE
tokenizer = Tokenizer(models.BPE())

# split teks jadi kata dasar
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=5000,   # bisa 2000–8000 untuk laptop
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
)

# training dari file kamu
files = ["data/input.txt"]

tokenizer.train(files, trainer)

# simpan
tokenizer.save("tokenizer.json")

print("Tokenizer berhasil dibuat")