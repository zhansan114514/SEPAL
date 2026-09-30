"""Continuation rollout rewards for ACC-Collab style DPO data."""

from __future__ import annotations

import copy
import statistics
from collections import Counter
from typing import Any

from src.evaluation.answer_resolution import answers_match
from src.paired import paired_critic_name
from src.paired.generation import (
    build_critic_prompts,
    build_revision_prompts,
    build_summary_prompts,
    make_response_record,
)
from src.paired.lora import generate_lora_batches
from src.parsing.think_blocks import clean_model_response
from src.prompts.actor_prompts import build_actor_guided_revision_prompt
from src.prompts.prompt_builder import build_actor_prompt, build_critic_prompt, build_problem_text

def _usable_record(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    if record.get("truncated"):
        return False
    return bool(str(record.get("response") or "").strip())


def _usable_text(text: str | None) -> bool:
    return bool(str(text or "").strip())


def _reward_delta_passes(delta: float, epsilon: float) -> bool:
    return delta > 0.0 and delta >= epsilon


def response_reward(
    response_record: dict[str, Any],
    sample: dict[str, Any],
) -> float:
    return float(
        answers_match(
            response_record.get("answer"),
            sample.get("answer"),
            sample.get("task_type", "multiple_choice"),
        )
    )


def mean_reward(records: list[dict[str, Any]], sample: dict[str, Any]) -> float:
    if not records:
        return 0.0
    return sum(response_reward(record, sample) for record in records) / len(records)


def make_rewarded_pair(
    *,
    prompt: str,
    chosen: str,
    rejected: str,
    chosen_reward: float,
    rejected_reward: float,
    record: dict[str, Any],
    actor_name: str,
    case_type: str,
    reward_rollouts: int,
    reward_final_rounds: int,
    round_index: int | None = None,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {
            "sample_id": record.get("sample_id"),
            "actor_name": actor_name,
            "critic_name": paired_critic_name(actor_name),
            "case_type": case_type,
            "round": round_index,
            "gold_answer": record.get("sample", {}).get("answer"),
            "reward_chosen": chosen_reward,
            "reward_rejected": rejected_reward,
            "reward_delta": chosen_reward - rejected_reward,
            "reward_rollouts": reward_rollouts,
            "reward_final_rounds": reward_final_rounds,
        },
    }


def _rounds(record: dict[str, Any]):
    """Yield ``(round_index, trajectory)`` for each collected deliberation round.

    ACC-Collab (Algorithm 1) collects guided-trajectory pairs at every round
    t in [1, T], not just round 1. Multi-round records store a ``trajectories``
    list; legacy single-round records (flat fields, no ``trajectories``) are
    yielded once as a synthetic round 0 so this stays backward compatible.
    """
    trajectories = record.get("trajectories")
    if trajectories:
        for index, trajectory in enumerate(trajectories):
            yield index, trajectory
        return
    yield 0, record


def build_critic_pairs_from_rewarded_records(
    records: list[dict[str, Any]],
    *,
    actor_names: list[str],
    dataset_name: str,
    reward_epsilon: float,
    reward_rollouts: int,
    reward_final_rounds: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    critic_pairs = {paired_critic_name(name): [] for name in actor_names}
    case_counts: Counter[str] = Counter()

    for record in records:
        sample = record.get("sample", {})
        prev_state = record.get("initial", {})
        for round_index, trajectory in _rounds(record):
            summaries = trajectory.get("summaries", {})
            natural_critics = trajectory.get("natural_critics", {})
            guided_pos_critics = trajectory.get("guided_positive_critics", {})
            guided_neg_critics = trajectory.get("guided_negative_critics", {})
            rewards = trajectory.get("critic_rewards", {})

            for actor_name in actor_names:
                init = prev_state.get(actor_name, {})
                nat_crit = str(natural_critics.get(actor_name) or "").strip()
                gp_crit = str(guided_pos_critics.get(actor_name) or "").strip()
                gn_crit = str(guided_neg_critics.get(actor_name) or "").strip()
                if not _usable_record(init) or not _usable_text(nat_crit):
                    continue

                r_nat = float(rewards.get(actor_name, {}).get("natural", 0.0))
                r_gp = float(rewards.get(actor_name, {}).get("guided_positive", 0.0))
                r_gn = float(rewards.get(actor_name, {}).get("guided_negative", 0.0))

                critic_name = paired_critic_name(actor_name)
                critic_prompt = build_critic_prompt(
                    critic_name,
                    sample,
                    dataset_name,
                    actor_name,
                    init.get("response", ""),
                    summaries.get(actor_name, ""),
                )

                if _usable_text(gp_crit) and _reward_delta_passes(r_gp - r_nat, reward_epsilon):
                    critic_pairs[critic_name].append(make_rewarded_pair(
                        prompt=critic_prompt,
                        chosen=gp_crit,
                        rejected=nat_crit,
                        chosen_reward=r_gp,
                        rejected_reward=r_nat,
                        record=record,
                        actor_name=actor_name,
                        case_type="critic_dy",
                        reward_rollouts=reward_rollouts,
                        reward_final_rounds=reward_final_rounds,
                        round_index=round_index,
                    ))
                    case_counts["critic_dy"] += 1

                if _usable_text(gn_crit) and _reward_delta_passes(r_nat - r_gn, reward_epsilon):
                    critic_pairs[critic_name].append(make_rewarded_pair(
                        prompt=critic_prompt,
                        chosen=nat_crit,
                        rejected=gn_crit,
                        chosen_reward=r_nat,
                        rejected_reward=r_gn,
                        record=record,
                        actor_name=actor_name,
                        case_type="critic_dny",
                        reward_rollouts=reward_rollouts,
                        reward_final_rounds=reward_final_rounds,
                        round_index=round_index,
                    ))
                    case_counts["critic_dny"] += 1

            # Advance the natural spine: the actor revisions produced under the
            # natural critic at this round become round t+1's prior state.
            prev_state = trajectory.get("state", prev_state)

    return critic_pairs, _pair_metrics({}, critic_pairs, records, case_counts)


def build_actor_pairs_from_rewarded_records(
    records: list[dict[str, Any]],
    *,
    actor_names: list[str],
    dataset_name: str,
    reward_epsilon: float,
    reward_rollouts: int,
    reward_final_rounds: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    actor_pairs = {name: [] for name in actor_names}
    case_counts: Counter[str] = Counter()

    for record in records:
        sample = record.get("sample", {})
        prev_state = record.get("initial", {})
        for round_index, trajectory in _rounds(record):
            natural_critics = trajectory.get("natural_critics", {})
            natural_revised = trajectory.get("natural_revised", {})
            guided_pos_revised = trajectory.get("guided_positive_revised", {})
            guided_neg_revised = trajectory.get("guided_negative_revised", {})
            rewards = trajectory.get("actor_rewards", {})

            for actor_name in actor_names:
                init = prev_state.get(actor_name, {})
                nat_rev = natural_revised.get(actor_name, {})
                gp_rev = guided_pos_revised.get(actor_name, {})
                gn_rev = guided_neg_revised.get(actor_name, {})
                nat_crit = natural_critics.get(actor_name, "")
                if not _usable_record(init) or not _usable_record(nat_rev):
                    continue

                r_nat = float(rewards.get(actor_name, {}).get("natural", 0.0))
                r_gp = float(rewards.get(actor_name, {}).get("guided_positive", 0.0))
                r_gn = float(rewards.get(actor_name, {}).get("guided_negative", 0.0))

                actor_prompt = build_actor_prompt(
                    actor_name,
                    sample,
                    dataset_name,
                    previous_response=init.get("response", ""),
                    critic_feedback=nat_crit,
                )

                if _usable_record(gp_rev) and _reward_delta_passes(r_gp - r_nat, reward_epsilon):
                    actor_pairs[actor_name].append(make_rewarded_pair(
                        prompt=actor_prompt,
                        chosen=gp_rev.get("response", ""),
                        rejected=nat_rev.get("response", ""),
                        chosen_reward=r_gp,
                        rejected_reward=r_nat,
                        record=record,
                        actor_name=actor_name,
                        case_type="actor_dy",
                        reward_rollouts=reward_rollouts,
                        reward_final_rounds=reward_final_rounds,
                        round_index=round_index,
                    ))
                    case_counts["actor_dy"] += 1

                if _usable_record(gn_rev) and _reward_delta_passes(r_nat - r_gn, reward_epsilon):
                    actor_pairs[actor_name].append(make_rewarded_pair(
                        prompt=actor_prompt,
                        chosen=nat_rev.get("response", ""),
                        rejected=gn_rev.get("response", ""),
                        chosen_reward=r_nat,
                        rejected_reward=r_gn,
                        record=record,
                        actor_name=actor_name,
                        case_type="actor_dny",
                        reward_rollouts=reward_rollouts,
                        reward_final_rounds=reward_final_rounds,
                        round_index=round_index,
                    ))
                    case_counts["actor_dny"] += 1

            # Advance the natural spine with this round's natural revisions.
            prev_state = natural_revised or prev_state

    return actor_pairs, _pair_metrics(actor_pairs, {}, records, case_counts)


def build_actor_guided_prompt_from_sample(
    *,
    actor_name: str,
    sample: dict[str, Any],
    dataset_name: str,
    previous_response: str,
    critic_feedback: str,
    target_answer: str,
) -> str:
    return build_actor_guided_revision_prompt(
        actor_name,
        build_problem_text(sample, dataset_name),
        previous_response,
        critic_feedback,
        target_answer,
    )


def make_generated_record(raw: str, sample: dict[str, Any]) -> dict[str, Any]:
    return make_response_record(
        raw_response=raw,
        task_type=sample.get("task_type", "multiple_choice"),
        sample=sample,
    )


def _reward_summary(pairs_by_agent: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate reward-margin diagnostics across built preference pairs.

    Surfaces the health of the continuation-reward signal: a tiny ``delta_mean``
    or a low ``robust_fraction_delta_ge_0p5`` means most pairs are separated by
    Monte-Carlo noise rather than a genuine quality gap, i.e. the DPO labels are
    weak and the positive-feedback loop is unlikely to improve accuracy.
    """
    deltas: list[float] = []
    chosen_rewards: list[float] = []
    rejected_rewards: list[float] = []
    for pairs in pairs_by_agent.values():
        for pair in pairs:
            meta = pair.get("metadata", {})
            delta = meta.get("reward_delta")
            if delta is None:
                continue
            deltas.append(float(delta))
            chosen_rewards.append(float(meta.get("reward_chosen", 0.0)))
            rejected_rewards.append(float(meta.get("reward_rejected", 0.0)))
    if not deltas:
        return {"count": 0}
    count = len(deltas)
    return {
        "count": count,
        "delta_mean": round(sum(deltas) / count, 4),
        "delta_median": round(statistics.median(deltas), 4),
        "delta_min": round(min(deltas), 4),
        "delta_max": round(max(deltas), 4),
        "chosen_reward_mean": round(sum(chosen_rewards) / count, 4),
        "rejected_reward_mean": round(sum(rejected_rewards) / count, 4),
        "robust_fraction_delta_ge_0p5": round(
            sum(1 for d in deltas if d >= 0.5) / count, 4
        ),
    }


def _pair_metrics(
    actor_pairs: dict[str, list[dict[str, Any]]],
    critic_pairs: dict[str, list[dict[str, Any]]],
    records: list[dict[str, Any]],
    case_counts: Counter[str],
) -> dict[str, Any]:
    return {
        "records": len(records),
        "actor_pair_counts": {name: len(pairs) for name, pairs in actor_pairs.items()},
        "critic_pair_counts": {name: len(pairs) for name, pairs in critic_pairs.items()},
        "case_counts": dict(case_counts),
        "reward_summary": _reward_summary({**actor_pairs, **critic_pairs}),
    }


# ---------------------------------------------------------------------------
# Batch reward estimation — cross-sample batching for vLLM efficiency
# ---------------------------------------------------------------------------


def _batch_advance_full_round(
    *,
    threads: list[dict[str, Any]],
    states: list[dict[str, dict[str, Any]]],
    actor_names: list[str],
    dataset_name: str,
    actor_adapters: dict[str, Any],
    critic_adapters: dict[str, Any],
    base_adapter: Any,
    actor_max_tokens: int,
    critic_max_tokens: int,
    summary_max_tokens: int,
    temperature: float,
    summary_temperature: float,
    enable_thinking: bool,
    top_p: float = 0.9,
) -> list[dict[str, dict[str, Any]]]:
    """Advance every thread by one deliberation round, batching all generation.

    Each thread may carry ``forced_feedback`` (str, critic mode) or
    ``forced_response`` (dict, actor mode).  When set, the target actor's
    critic generation is skipped (forced value injected afterwards) and,
    for actor mode, the target actor's revision is also skipped.
    """
    n = len(threads)

    # --- Step 1: Summaries (base model, one batch) ---
    summary_prompts_flat: list[str] = []
    summary_indices: list[tuple[int, str]] = []
    for i, thread in enumerate(threads):
        prompts = build_summary_prompts(
            actor_names, thread["sample"], dataset_name, states[i],
        )
        for name in actor_names:
            summary_prompts_flat.append(prompts[name])
            summary_indices.append((i, name))

    # Deduplicate identical summary prompts.  In round 1, variants of the
    # same (sample, actor) share initial_state → identical prompt.  This is
    # safe because summary_temperature is 0.0 (deterministic decoding).
    _unique: dict[str, int] = {}
    _uniq_map: list[int] = []
    for p in summary_prompts_flat:
        idx = _unique.get(p)
        if idx is None:
            idx = len(_unique)
            _unique[p] = idx
        _uniq_map.append(idx)

    _unique_list = list(_unique.keys())
    _unique_outputs = base_adapter.generate(
        _unique_list,
        max_tokens=summary_max_tokens,
        temperature=summary_temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
    )
    summary_outputs = [_unique_outputs[_uniq_map[k]] for k in range(len(summary_prompts_flat))]

    summaries: list[dict[str, str]] = [{} for _ in range(n)]
    for (ti, name), raw in zip(summary_indices, summary_outputs):
        summaries[ti][name] = clean_model_response(raw)

    # --- Step 2: Critic feedbacks (LoRA, one batch per round) ---
    critic_prompts_by_actor: dict[str, list[str]] = {name: [] for name in actor_names}
    critic_thread_idx: dict[str, list[int]] = {name: [] for name in actor_names}

    for i, thread in enumerate(threads):
        sample = thread["sample"]
        target_actor = thread["actor_name"]
        target_critic = thread["critic_name"]
        forced_feedback = thread.get("forced_feedback")
        forced_response = thread.get("forced_response")

        for name in actor_names:
            # Skip target actor when forced (either mode)
            if name == target_actor and (
                forced_feedback is not None or forced_response is not None
            ):
                continue
            critic_name = (
                target_critic if name == target_actor else paired_critic_name(name)
            )
            prompt = build_critic_prompts(
                [name],
                {name: critic_name},
                sample,
                dataset_name,
                {name: states[i][name]},
                {name: summaries[i][name]},
            )[name]
            critic_prompts_by_actor[name].append(prompt)
            critic_thread_idx[name].append(i)

    critic_outputs_by_actor = generate_lora_batches(
        critic_adapters,
        critic_prompts_by_actor,
        max_tokens=critic_max_tokens,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
    )

    feedbacks: list[dict[str, str]] = [{} for _ in range(n)]
    for name in actor_names:
        for ti, raw in zip(critic_thread_idx[name], critic_outputs_by_actor[name]):
            feedbacks[ti][name] = clean_model_response(raw)

    # Inject forced critic feedback (critic mode)
    for i, thread in enumerate(threads):
        forced_feedback = thread.get("forced_feedback")
        if forced_feedback is not None:
            feedbacks[i][thread["actor_name"]] = forced_feedback

    # --- Step 3: Revisions (LoRA, one batch per round) ---
    revision_prompts_by_actor: dict[str, list[str]] = {name: [] for name in actor_names}
    revision_thread_idx: dict[str, list[int]] = {name: [] for name in actor_names}

    for i, thread in enumerate(threads):
        sample = thread["sample"]
        target_actor = thread["actor_name"]
        forced_response = thread.get("forced_response")

        for name in actor_names:
            if name == target_actor and forced_response is not None:
                continue
            prompt = build_revision_prompts(
                [name],
                sample,
                dataset_name,
                {name: states[i][name]},
                {name: feedbacks[i].get(name, "")},
            )[name]
            revision_prompts_by_actor[name].append(prompt)
            revision_thread_idx[name].append(i)

    revision_outputs_by_actor = generate_lora_batches(
        actor_adapters,
        revision_prompts_by_actor,
        max_tokens=actor_max_tokens,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
    )

    next_states: list[dict[str, dict[str, Any]]] = [{} for _ in range(n)]
    for name in actor_names:
        for ti, raw in zip(revision_thread_idx[name], revision_outputs_by_actor[name]):
            next_states[ti][name] = make_generated_record(raw, threads[ti]["sample"])

    # Inject forced response record (actor mode)
    for i, thread in enumerate(threads):
        forced_response = thread.get("forced_response")
        if forced_response is not None:
            next_states[i][thread["actor_name"]] = dict(forced_response)

    return next_states


def batch_estimate_reward(
    *,
    threads: list[dict[str, Any]],
    actor_names: list[str],
    dataset_name: str,
    actor_adapters: dict[str, Any],
    critic_adapters: dict[str, Any],
    base_adapter: Any,
    actor_max_tokens: int,
    critic_max_tokens: int,
    summary_max_tokens: int,
    temperature: float,
    summary_temperature: float,
    reward_rollouts: int,
    reward_final_rounds: int,
    enable_thinking: bool,
    top_p: float = 0.9,
) -> list[float]:
    """Batch reward estimation for multiple (sample, actor, variant) threads.

    Each thread dict must contain:
        sample, actor_name, critic_name, initial_state.

    Each thread's first-round behavior is determined by whether it carries
    ``forced_feedback`` (str, Critic feedback candidate) or ``forced_response``
    (dict, Actor response candidate).

    Returns one float per thread (mean reward across rollouts).

    Round structure (ACC-Collab §4.2 one-step roll-out): the candidate is forced
    into round 1, then ``reward_final_rounds - 2`` natural continuation rounds are
    simulated; the reward is the mean final-answer correctness over
    ``reward_rollouts`` independent rollouts. ``reward_final_rounds`` must be >= 3
    so the candidate is propagated by at least one natural continuation round;
    at 2 the reward degenerates to the candidate's own correctness.
    """
    if reward_final_rounds < 3:
        raise ValueError(
            "reward_final_rounds must be >= 3: the forced candidate occupies round "
            "1 and needs >= 1 natural continuation round (reward_final_rounds - 2) "
            "to be a genuine ACC-Collab one-step roll-out."
        )

    n = len(threads)
    final_records: list[list[dict[str, Any]]] = [[] for _ in range(n)]

    for _ in range(reward_rollouts):
        states: list[dict[str, dict[str, Any]]] = [
            copy.deepcopy(t["initial_state"]) for t in threads
        ]

        # Round 1: forced candidate
        states = _batch_advance_full_round(
            threads=threads,
            states=states,
            actor_names=actor_names,
            dataset_name=dataset_name,
            actor_adapters=actor_adapters,
            critic_adapters=critic_adapters,
            base_adapter=base_adapter,
            actor_max_tokens=actor_max_tokens,
            critic_max_tokens=critic_max_tokens,
            summary_max_tokens=summary_max_tokens,
            temperature=temperature,
            summary_temperature=summary_temperature,
            enable_thinking=enable_thinking,
            top_p=top_p,
        )

        # Rounds 2 to reward_final_rounds-1: natural (clear forced values)
        natural_threads = [
            {**t, "forced_feedback": None, "forced_response": None} for t in threads
        ]
        for _ in range(reward_final_rounds - 2):
            states = _batch_advance_full_round(
                threads=natural_threads,
                states=states,
                actor_names=actor_names,
                dataset_name=dataset_name,
                actor_adapters=actor_adapters,
                critic_adapters=critic_adapters,
                base_adapter=base_adapter,
                actor_max_tokens=actor_max_tokens,
                critic_max_tokens=critic_max_tokens,
                summary_max_tokens=summary_max_tokens,
                temperature=temperature,
                summary_temperature=summary_temperature,
                enable_thinking=enable_thinking,
                top_p=top_p,
            )

        for i, thread in enumerate(threads):
            final_records[i].append(states[i][thread["actor_name"]])

    return [
        mean_reward(final_records[i], threads[i]["sample"])
        for i in range(n)
    ]
