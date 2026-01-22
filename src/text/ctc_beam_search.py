from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import torch


def _log_sum_exp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


@dataclass
class BeamEntry:
    p_b: float   # log prob ending with blank
    p_nb: float  # log prob ending with non-blank


def ctc_prefix_beam_search(
    log_probs: torch.Tensor,  # [T, V] log-probs
    beam_size: int = 10,
    blank_id: int = 0,
) -> List[int]:
    T, V = log_probs.shape
    beam: Dict[Tuple[int, ...], BeamEntry] = {(): BeamEntry(p_b=0.0, p_nb=-math.inf)}

    for t in range(T):
        next_beam: Dict[Tuple[int, ...], BeamEntry] = {}

        def get_entry(prefix: Tuple[int, ...]) -> BeamEntry:
            e = next_beam.get(prefix)
            if e is None:
                e = BeamEntry(p_b=-math.inf, p_nb=-math.inf)
                next_beam[prefix] = e
            return e

        for prefix, entry in beam.items():
            p_b, p_nb = entry.p_b, entry.p_nb

            # blank
            p_blank = float(log_probs[t, blank_id].item())
            e_same = get_entry(prefix)
            e_same.p_b = _log_sum_exp(e_same.p_b, _log_sum_exp(p_b + p_blank, p_nb + p_blank))

            last = prefix[-1] if len(prefix) > 0 else None

            # non-blanks
            for c in range(V):
                if c == blank_id:
                    continue
                p_c = float(log_probs[t, c].item())

                if last is not None and c == last:
                    # from non-blank staying in same prefix
                    e1 = get_entry(prefix)
                    e1.p_nb = _log_sum_exp(e1.p_nb, p_nb + p_c)

                    # from blank -> extend
                    new_pref = prefix + (c,)
                    e2 = get_entry(new_pref)
                    e2.p_nb = _log_sum_exp(e2.p_nb, p_b + p_c)
                else:
                    new_pref = prefix + (c,)
                    e2 = get_entry(new_pref)
                    e2.p_nb = _log_sum_exp(e2.p_nb, _log_sum_exp(p_b + p_c, p_nb + p_c))

        scored = []
        for pref, e in next_beam.items():
            total = _log_sum_exp(e.p_b, e.p_nb)
            scored.append((total, pref))
        scored.sort(key=lambda x: x[0], reverse=True)

        beam = {}
        for _, pref in scored[: max(1, int(beam_size))]:
            beam[pref] = next_beam[pref]

    best_pref = max(beam.items(), key=lambda kv: _log_sum_exp(kv[1].p_b, kv[1].p_nb))[0]
    return list(best_pref)
