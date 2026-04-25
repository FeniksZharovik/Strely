import os

input_file = "data/train_split.txt"
output_file = "data/input.txt"

max_chars = 10_000_000  
collected = 0

with open(input_file, "r", encoding="utf-8", errors="ignore") as f, \
     open(output_file, "w", encoding="utf-8") as out:

    for line in f:
        if collected >= max_chars:
            break
        
        out.write(line)
        collected += len(line)

print("Sample dataset berhasil dibuat (±10MB)")