# Corpus preparation

The harness works out of the box with built-in sample sentences (~10 lines per
script) which is enough to smoke-test the pipeline.  For Gate A/B you need
≥150 lines per script (200 recommended).  Use the sources below.

---

## Tamil (`corpora/tamil.txt`)

**Recommended: AI4Bharat IndicCorp v2**
```
# From https://huggingface.co/datasets/ai4bharat/IndicCorp
# Filter language == 'ta', take first 500 sentences, write one sentence per line.
from datasets import load_dataset
ds = load_dataset("ai4bharat/IndicCorp", "ta", split="train", streaming=True)
with open("corpora/tamil.txt", "w") as f:
    for i, ex in enumerate(ds):
        if i >= 500: break
        line = ex["text"].strip().replace("\n", " ")
        if 20 < len(line) < 200:  # reasonable line length
            f.write(line + "\n")
```

**Alternative: Tamil Wikipedia**
```
pip install datasets
from datasets import load_dataset
ds = load_dataset("wikipedia", "20231101.ta", split="train", streaming=True)
# Extract first sentence of each article.
```

**Alternative: Project Madurai (permissive)**
Download plain-text files from https://www.projectmadurai.org/
and split into single lines.

---

## Devanagari (`corpora/devanagari.txt`)

**AI4Bharat IndicCorp, language='hi'** (same approach as Tamil above).
Hindi Wikipedia also works.

---

## Latin (`corpora/latin.txt`)

Any English newline-separated text works.  Wikipedia dump or
```python
from datasets import load_dataset
ds = load_dataset("wikipedia", "20231101.en", split="train", streaming=True)
```

---

## Format

Each file: one sentence per line, UTF-8, no blank lines.
Aim for 20–200 chars per line (longer lines produce wide images that stress the
compression budget more, which is good for Pillar 3 signal).

---

## Font files

Place in `fonts/`:
- `NotoSansTamil-Regular.ttf`   — https://fonts.google.com/noto/specimen/Noto+Sans+Tamil
- `NotoSansDevanagari-Regular.ttf` — https://fonts.google.com/noto/specimen/Noto+Sans+Devanagari
- `NotoSans-Regular.ttf`        — https://fonts.google.com/noto/specimen/Noto+Sans

All are OFL 1.1 licensed — safe for HuggingFace dataset release.

Quick CLI download (if `gfonts` is available):
```
pip install gfonts
gfonts download "Noto Sans Tamil" "Noto Sans Devanagari" "Noto Sans"
mv ~/.local/share/fonts/Noto*.ttf fonts/
```
Or download TTF files manually from the links above.
