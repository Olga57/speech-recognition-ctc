from __future__ import annotations
from jiwer import wer as _wer

def get_wer(ref: str, hyp: str) -> float:
    ref = (ref or "").strip().lower()
    hyp = (hyp or "").strip().lower()
    return float(_wer(ref, hyp))

def get_cer(ref: str, hyp: str) -> float:
    ref = (ref or "").strip().lower()
    hyp = (hyp or "").strip().lower()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    # Levenshtein on characters
    r = list(ref)
    h = list(hyp)
    dp = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1):
        dp[i][0] = i
    for j in range(len(h)+1):
        dp[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[len(r)][len(h)] / max(1, len(r))
