"""Metrics for paired evaluation and DPO data generation."""

from __future__ import annotations

from collections import Counter
from typing import Any


def summarize_generation_records(
    records: list[dict[str, Any]],
    actor_names: list[str],
) -> dict[str, Any]:
    """Summarize generation records built by the 8-step guided flow.

    Record shape (per sample):
      initial, natural_revised, guided_positive_revised, guided_negative_revised
        : dict[actor_name -> response_record]
      natural_critics, guided_positive_critics, guided_negative_critics
        : dict[actor_name -> str]
    """
    totals = Counter()

    for record in records:
        for actor_name in actor_names:
            init = record.get("initial", {}).get(actor_name, {})
            nat_rev = record.get("natural_revised", {}).get(actor_name, {})
            gp_rev = record.get("guided_positive_revised", {}).get(actor_name, {})
            gn_rev = record.get("guided_negative_revised", {}).get(actor_name, {})

            for field, rec in (
                ("initial", init),
                ("natural_revised", nat_rev),
                ("guided_positive_revised", gp_rev),
                ("guided_negative_revised", gn_rev),
            ):
                if not rec:
                    continue
                totals[f"{field}_total"] += 1
                totals[f"{field}_correct"] += int(bool(rec.get("correct")))
                if rec.get("truncated"):
                    totals[f"{field}_truncated"] += 1

            if init and nat_rev:
                totals["correct_to_wrong"] += int(
                    bool(init.get("correct")) and not bool(nat_rev.get("correct"))
                )
                totals["wrong_to_correct"] += int(
                    not bool(init.get("correct")) and bool(nat_rev.get("correct"))
                )
                totals["wrong_to_wrong"] += int(
                    not bool(init.get("correct")) and not bool(nat_rev.get("correct"))
                )

            # Actor-pair eligibility (ACC-Collab Eq. 5 with 0/1 reward proxy).
            r_nat = bool(nat_rev.get("correct")) if nat_rev else None
            r_gp = bool(gp_rev.get("correct")) if gp_rev else None
            r_gn = bool(gn_rev.get("correct")) if gn_rev else None

            if r_nat is not None and r_gp is not None:
                totals["actor_pair_eligible_total"] += 1
                if r_gp and not r_nat:
                    totals["actor_pair_dy"] += 1  # chosen=guided+, rejected=natural
            if r_nat is not None and r_gn is not None:
                totals["actor_pair_eligible_total"] += 1
                if r_nat and not r_gn:
                    totals["actor_pair_dny"] += 1  # chosen=natural, rejected=guided-

            # Critic pair is always available whenever natural_revised exists:
            #   if natural was wrong  -> (guided+ critic, natural critic)
            #   if natural was right  -> (natural critic, guided- critic)
            if nat_rev:
                totals["critic_pair_total"] += 1

    return {
        "actor_initial_accuracy": _rate(totals["initial_correct"], totals["initial_total"]),
        "natural_revised_accuracy": _rate(
            totals["natural_revised_correct"], totals["natural_revised_total"]
        ),
        "guided_positive_revised_accuracy": _rate(
            totals["guided_positive_revised_correct"], totals["guided_positive_revised_total"]
        ),
        "guided_negative_revised_accuracy": _rate(
            totals["guided_negative_revised_correct"], totals["guided_negative_revised_total"]
        ),
        "critic_correction_rate": _rate(
            totals["wrong_to_correct"], totals["wrong_to_correct"] + totals["wrong_to_wrong"]
        ),
        "harmful_flip_rate": _rate(totals["correct_to_wrong"], totals["initial_correct"]),
        "actor_pair_dy_rate": _rate(totals["actor_pair_dy"], totals["actor_pair_eligible_total"]),
        "actor_pair_dny_rate": _rate(totals["actor_pair_dny"], totals["actor_pair_eligible_total"]),
        "counts": dict(totals),
    }


def _rate(num: int, den: int) -> float:
    return num / den if den else 0.0
