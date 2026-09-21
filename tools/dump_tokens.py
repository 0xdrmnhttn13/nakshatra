#!/usr/bin/env python3
import argparse
import struct
import sys
from pathlib import Path


def load_tokenizer(model):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("pip install transformers first")

    try:
        return AutoTokenizer.from_pretrained(model)
    except Exception as e:
        sys.exit(f"cannot load tokenizer '{model}': {e}")


def escape_piece(piece):
    return (
        piece.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )


def main():
    ap = argparse.ArgumentParser(description="Gemma tokenizer bridge for nakshatra")
    ap.add_argument("prompt", nargs="?", help="prompt text to tokenize")
    ap.add_argument("--file", type=Path, help="read prompt from file")
    ap.add_argument("--model", default="google/gemma-3-1b-pt")
    ap.add_argument("--out", type=Path, help="write ids as little-endian u32 binary")
    ap.add_argument(
        "--export-vocab",
        type=Path,
        dest="vocab",
        help="write id<TAB>piece table for C++ decode",
    )
    args = ap.parse_args()

    tok = load_tokenizer(args.model)

    if args.vocab:
        vocab = tok.get_vocab()
        rows = sorted(vocab.items(), key=lambda kv: kv[1])
        with open(args.vocab, "w", encoding="utf-8") as f:
            for piece, idx in rows:
                f.write(f"{idx}\t{escape_piece(piece)}\n")
        print(f"wrote {len(rows)} vocab rows")

    prompt = None
    if args.file:
        prompt = args.file.read_text(encoding="utf-8")
    elif args.prompt is not None:
        prompt = args.prompt

    if prompt is not None:
        ids = tok(prompt)["input_ids"]
        print(" ".join(str(i) for i in ids))
        pieces = tok.convert_ids_to_tokens(ids)
        print("pieces: " + " ".join(pieces))

        if args.out:
            with open(args.out, "wb") as f:
                for i in ids:
                    f.write(struct.pack("<I", i))


if __name__ == "__main__":
    main()
