"""Multi-round guided-trajectory generation for ACC-Collab DPO data.

ACC-Collab (Algorithm 1) collects guided-trajectory preference pairs at *every*
deliberation round t in [1, T], not only round 1. The data scripts advance a
*natural spine* of responses round over round; at each round they branch
off-policy guided+ / guided- candidates, estimate their continuation reward,
and emit a per-round trajectory record consumed by
:func:`src.paired.rollout_reward.build_actor_pairs_from_rewarded_records` /
:func:`src.paired.rollout_reward.build_critic_pairs_from_rewarded_records`.

These helpers factor one deliberation round so the scripts become a short loop
and the round logic is unit-testable with fake adapters.
"""

from __future__ import annotations

from typing import Any

from src.paired.generation import (
    build_critic_prompts,
    build_revision_prompts,
    build_summary_prompts,
    make_response_record,
)
from src.paired.lora import generate_lora_batches
from src.paired.rollout_reward import build_actor_guided_prompt_from_sample
from src.parsing.think_blocks import clean_model_response
from src.prompts.critic_prompts import build_guided_critic_prompt
from src.prompts.prompt_builder import build_problem_text


def wrong_answer_for_sample(sample: dict[str, Any]) -> str:
    """A plausible incorrect target for guided- (away-from-correct) prompting."""
    task_type = sample.get("task_type", "multiple_choice")
    correct = str(sample.get("answer") or "").strip().upper()
    if task_type == "yes_no":
        return "NO" if correct in {"YES", "Y"} else "YES"
    if task_type == "multiple_choice":
        for option in "ABCD":
            if option != correct:
                return option
    return "an incorrect alternative"


def _generate_summaries(
    batch, current_state, names, dataset_name, base, summary_max_tokens,
    summary_temperature, top_p, enable_thinking,
):
    summary_prompts: list[str] = []
    summary_keys: list[tuple[int, str]] = []
    for i, sample in enumerate(batch):
        prompts = build_summary_prompts(names, sample, dataset_name, current_state[i])
        for name, prompt in prompts.items():
            summary_prompts.append(prompt)
            summary_keys.append((i, name))
    raw = base.generate(
        summary_prompts,
        max_tokens=summary_max_tokens,
        temperature=summary_temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
    )
    summaries: list[dict[str, str]] = [{} for _ in batch]
    for (i, name), text in zip(summary_keys, raw):
        summaries[i][name] = clean_model_response(text)
    return summaries


def generate_actor_trajectory_round(
    *,
    batch,
    prev_state,
    names,
    dataset_name,
    actor_adapters,
    critic_adapters,
    critic_names,
    base,
    actor_max_tokens,
    critic_max_tokens,
    summary_max_tokens,
    temperature,
    summary_temperature,
    top_p,
    enable_thinking,
    reward_rollouts,
    reward_final_rounds,
    round_index,
) -> list[tuple[dict[str, Any], dict[str, dict[str, Any]]]]:
    """Generate one actor-candidate round. Returns per-sample ``(trajectory, next_state)``.

    ``next_state`` is the natural-revision spine for round t+1. ``critic_adapters``
    are keyed by critic name. The reward of each candidate is estimated by a
    continuation roll-out rooted at ``prev_state`` (the round-t-1 spine).
    """
    from src.paired.rollout_reward import batch_estimate_reward

    summaries = _generate_summaries(
        batch, prev_state, names, dataset_name, base, summary_max_tokens,
        summary_temperature, top_p, enable_thinking,
    )

    # Natural critic feedback per actor. critic_adapters is keyed by actor name
    # (each entry is that actor's paired critic adapter), so the prompt dict is
    # keyed by actor name too — generate_lora_batches looks up the adapter by
    # this key.
    critic_prompts_by_actor: dict[str, list[str]] = {name: [] for name in names}
    for name in names:
        critic_prompts_by_actor[name] = [
            build_critic_prompts(
                [name], {name: critic_names[name]}, sample, dataset_name,
                {name: prev_state[i][name]}, {name: summaries[i][name]},
            )[name]
            for i, sample in enumerate(batch)
        ]
    critic_outputs = generate_lora_batches(
        critic_adapters, critic_prompts_by_actor,
        max_tokens=critic_max_tokens, temperature=temperature,
        top_p=top_p, enable_thinking=enable_thinking,
    )
    natural_critics: list[dict[str, str]] = [{} for _ in batch]
    for name, feedbacks in critic_outputs.items():
        for i, feedback in enumerate(feedbacks):
            natural_critics[i][name] = clean_model_response(feedback)

    # Natural + guided+ / guided- revisions.
    natural_prompts: dict[str, list[str]] = {name: [] for name in names}
    guided_prompts: dict[str, list[str]] = {name: [] for name in names}
    guided_keys: dict[str, list[tuple[int, str]]] = {name: [] for name in names}
    for i, sample in enumerate(batch):
        natural = build_revision_prompts(
            names, sample, dataset_name, prev_state[i], natural_critics[i],
        )
        correct_answer = str(sample.get("answer") or "").strip()
        wrong_answer = wrong_answer_for_sample(sample)
        for name in names:
            natural_prompts[name].append(natural[name])
            previous = prev_state[i][name].get("response", "")
            feedback = natural_critics[i][name]
            for sign, target in (("+", correct_answer), ("-", wrong_answer)):
                guided_prompts[name].append(
                    build_actor_guided_prompt_from_sample(
                        actor_name=name, sample=sample, dataset_name=dataset_name,
                        previous_response=previous, critic_feedback=feedback,
                        target_answer=target,
                    )
                )
                guided_keys[name].append((i, sign))

    revision_prompts = {
        name: natural_prompts[name] + guided_prompts[name] for name in names
    }
    revision_outputs = generate_lora_batches(
        actor_adapters, revision_prompts,
        max_tokens=actor_max_tokens, temperature=temperature,
        top_p=top_p, enable_thinking=enable_thinking,
    )
    natural_revised: list[dict[str, dict[str, Any]]] = [{} for _ in batch]
    guided_pos_revised: list[dict[str, dict[str, Any]]] = [{} for _ in batch]
    guided_neg_revised: list[dict[str, dict[str, Any]]] = [{} for _ in batch]
    for name in names:
        outputs = revision_outputs[name]
        for i, sample in enumerate(batch):
            natural_revised[i][name] = make_response_record(
                raw_response=outputs[i],
                task_type=sample.get("task_type", "multiple_choice"),
                sample=sample,
            )
        for output_index, (sample_index, sign) in enumerate(
            guided_keys[name], start=len(batch)
        ):
            sample = batch[sample_index]
            record = make_response_record(
                raw_response=outputs[output_index],
                task_type=sample.get("task_type", "multiple_choice"),
                sample=sample,
            )
            if sign == "+":
                guided_pos_revised[sample_index][name] = record
            else:
                guided_neg_revised[sample_index][name] = record

    # Continuation reward for each candidate, rooted at the round-t-1 spine.
    reward_threads: list[dict[str, Any]] = []
    thread_map: list[tuple[int, str, str]] = []
    for i, sample in enumerate(batch):
        for name in names:
            for variant, revised in (
                ("natural", natural_revised[i]),
                ("guided_positive", guided_pos_revised[i]),
                ("guided_negative", guided_neg_revised[i]),
            ):
                reward_threads.append({
                    "sample": sample,
                    "actor_name": name,
                    "critic_name": critic_names[name],
                    "forced_response": revised[name],
                    "initial_state": prev_state[i],
                })
                thread_map.append((i, name, variant))
    reward_values = batch_estimate_reward(
        threads=reward_threads, actor_names=names, dataset_name=dataset_name,
        actor_adapters=actor_adapters, critic_adapters=critic_adapters,
        base_adapter=base, actor_max_tokens=actor_max_tokens,
        critic_max_tokens=critic_max_tokens, summary_max_tokens=summary_max_tokens,
        temperature=temperature, summary_temperature=summary_temperature,
        reward_rollouts=reward_rollouts, reward_final_rounds=reward_final_rounds,
        enable_thinking=enable_thinking, top_p=top_p,
    )
    actor_rewards: list[dict[str, dict[str, float]]] = [{} for _ in batch]
    for (i, name, variant), reward in zip(thread_map, reward_values):
        actor_rewards[i].setdefault(name, {})[variant] = reward

    results = []
    for i in range(len(batch)):
        trajectory = {
            "round": round_index,
            "summaries": summaries[i],
            "natural_critics": natural_critics[i],
            "natural_revised": natural_revised[i],
            "guided_positive_revised": guided_pos_revised[i],
            "guided_negative_revised": guided_neg_revised[i],
            "actor_rewards": actor_rewards[i],
        }
        results.append((trajectory, natural_revised[i]))
    return results


def generate_critic_trajectory_round(
    *,
    batch,
    prev_state,
    names,
    dataset_name,
    actor_adapters,
    critic_adapters,
    critic_names,
    base,
    actor_max_tokens,
    critic_max_tokens,
    summary_max_tokens,
    temperature,
    summary_temperature,
    top_p,
    enable_thinking,
    reward_rollouts,
    reward_final_rounds,
    round_index,
    advance_spine: bool,
) -> list[tuple[dict[str, Any], dict[str, dict[str, Any]]]]:
    """Generate one critic-candidate round. Returns per-sample ``(trajectory, next_state)``.

    ``critic_adapters`` are keyed by actor name (the adapter playing each actor's
    critic — the actor-SFT adapter during critic bootstrapping). When
    ``advance_spine`` is set, the actor's natural revision under the natural
    critic is generated to become round t+1's spine (stored as ``state``); this
    is skipped on the final round to avoid wasted generation.
    """
    from src.paired.rollout_reward import batch_estimate_reward

    summaries = _generate_summaries(
        batch, prev_state, names, dataset_name, base, summary_max_tokens,
        summary_temperature, top_p, enable_thinking,
    )

    # Natural critic feedback (critic adapter keyed by actor name).
    natural_prompts: dict[str, list[str]] = {name: [] for name in names}
    for i, sample in enumerate(batch):
        prompts = build_critic_prompts(
            names, critic_names, sample, dataset_name, prev_state[i], summaries[i],
        )
        for name, prompt in prompts.items():
            natural_prompts[name].append(prompt)
    critic_outputs = generate_lora_batches(
        critic_adapters, natural_prompts,
        max_tokens=critic_max_tokens, temperature=temperature,
        top_p=top_p, enable_thinking=enable_thinking,
    )
    natural_critics: list[dict[str, str]] = [{} for _ in batch]
    for name, feedbacks in critic_outputs.items():
        for i, feedback in enumerate(feedbacks):
            natural_critics[i][name] = clean_model_response(feedback)

    # Guided+ / guided- critic feedback from the base model.
    guided_prompts: list[str] = []
    guided_keys: list[tuple[int, str, str]] = []
    for i, sample in enumerate(batch):
        problem_text = build_problem_text(sample, dataset_name)
        correct_answer = str(sample.get("answer") or "").strip()
        wrong_answer = wrong_answer_for_sample(sample)
        for name in names:
            init_resp = prev_state[i][name].get("response", "")
            peer_summary = summaries[i].get(name, "")
            for sign, target in (("+", correct_answer), ("-", wrong_answer)):
                guided_prompts.append(
                    build_guided_critic_prompt(
                        critic_names[name], problem_text, name, init_resp,
                        peer_summary, target,
                    )
                )
                guided_keys.append((i, name, sign))
    guided_outputs = base.generate(
        guided_prompts,
        max_tokens=critic_max_tokens, temperature=temperature,
        top_p=top_p, enable_thinking=enable_thinking,
    )
    guided_pos_critics: list[dict[str, str]] = [{} for _ in batch]
    guided_neg_critics: list[dict[str, str]] = [{} for _ in batch]
    for (i, name, sign), feedback in zip(guided_keys, guided_outputs):
        target = guided_pos_critics if sign == "+" else guided_neg_critics
        target[i][name] = clean_model_response(feedback)

    # Continuation reward for each critic candidate.
    reward_threads: list[dict[str, Any]] = []
    thread_map: list[tuple[int, str, str]] = []
    for i, sample in enumerate(batch):
        for name in names:
            for variant, critic_store in (
                ("natural", natural_critics[i]),
                ("guided_positive", guided_pos_critics[i]),
                ("guided_negative", guided_neg_critics[i]),
            ):
                reward_threads.append({
                    "sample": sample,
                    "actor_name": name,
                    "critic_name": critic_names[name],
                    "forced_feedback": critic_store.get(name, ""),
                    "initial_state": prev_state[i],
                })
                thread_map.append((i, name, variant))
    reward_values = batch_estimate_reward(
        threads=reward_threads, actor_names=names, dataset_name=dataset_name,
        actor_adapters=actor_adapters, critic_adapters=critic_adapters,
        base_adapter=base, actor_max_tokens=actor_max_tokens,
        critic_max_tokens=critic_max_tokens, summary_max_tokens=summary_max_tokens,
        temperature=temperature, summary_temperature=summary_temperature,
        reward_rollouts=reward_rollouts, reward_final_rounds=reward_final_rounds,
        enable_thinking=enable_thinking, top_p=top_p,
    )
    critic_rewards: list[dict[str, dict[str, float]]] = [{} for _ in batch]
    for (i, name, variant), reward in zip(thread_map, reward_values):
        critic_rewards[i].setdefault(name, {})[variant] = reward

    # Optional spine advance: actor revises under the natural critic.
    next_states: list[dict[str, dict[str, Any]]] = list(prev_state)
    if advance_spine:
        revision_prompts: dict[str, list[str]] = {name: [] for name in names}
        for i, sample in enumerate(batch):
            prompts = build_revision_prompts(
                names, sample, dataset_name, prev_state[i], natural_critics[i],
            )
            for name, prompt in prompts.items():
                revision_prompts[name].append(prompt)
        revision_outputs = generate_lora_batches(
            actor_adapters, revision_prompts,
            max_tokens=actor_max_tokens, temperature=temperature,
            top_p=top_p, enable_thinking=enable_thinking,
        )
        next_states = [{} for _ in batch]
        for name, outputs in revision_outputs.items():
            for i, sample in enumerate(batch):
                next_states[i][name] = make_response_record(
                    raw_response=outputs[i],
                    task_type=sample.get("task_type", "multiple_choice"),
                    sample=sample,
                )

    results = []
    for i in range(len(batch)):
        trajectory = {
            "round": round_index,
            "summaries": summaries[i],
            "natural_critics": natural_critics[i],
            "guided_positive_critics": guided_pos_critics[i],
            "guided_negative_critics": guided_neg_critics[i],
            "critic_rewards": critic_rewards[i],
            "state": next_states[i],
        }
        results.append((trajectory, next_states[i]))
    return results
