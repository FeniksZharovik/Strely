import glob

files = glob.glob("data/*.txt")

max_chars = 5_000_000  # batasi total karakter (biar aman)
collected = 0

with open("data/input.txt", "w", encoding="utf-8") as out:
    for file in files:
        print(f"Processing {file}")
        
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if collected >= max_chars:
                    break
                
                out.write(line.strip() + "\n")
                collected += len(line)

        if collected >= max_chars:
            break

print("Selesai tanpa overload RAM")