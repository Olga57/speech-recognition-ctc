from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class CharTokenizer:

    vocab: str = " abcdefghijklmnopqrstuvwxyz'"

    def __post_init__(self):
        self.blank = 0
        self.id2ch: Dict[int, str] = {0: "<blank>"}
        self.ch2id: Dict[str, int] = {}
        for i, ch in enumerate(self.vocab, start=1):
            self.id2ch[i] = ch
            self.ch2id[ch] = i

    @property
    def vocab_size(self) -> int:
        return 1 + len(self.vocab)

    def encode(self, text: str) -> List[int]:
        text = (text or "").lower()
        out = []
        for ch in text:
            if ch in self.ch2id:
                out.append(self.ch2id[ch])

        return out

    def decode(self, ids: List[int]) -> str:
        chars = []
        for i in ids:
            if i == self.blank:
                continue
            ch = self.id2ch.get(int(i), "")
            if ch and ch != "<blank>":
                chars.append(ch)

        s = "".join(chars)
        s = " ".join(s.split())
        return s
