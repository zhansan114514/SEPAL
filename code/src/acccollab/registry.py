"""Policy-state resolution and adapter registries for isolated ACC-Collab runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.acccollab.config import ACCCollabConfig, ACCCollabPaths
from src.acccollab.io import read_json, write_json
from src.utils.artifacts import completed_adapter_path, path_identity


class ACCCollabRegistryError(RuntimeError):
    """Raised when an alternating-training policy state is incomplete."""


@dataclass(frozen=True)
class PolicyState:
    """Actor/Critic adapters active at one data-generation or evaluation stage."""

    actor_adapter: str | None
    critic_adapter: str | None
    actor_iteration: int
    critic_iteration: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["actor_policy"] = self.actor_adapter or "base_model"
        payload["critic_policy"] = self.critic_adapter or "base_model"
        return payload


def _require_adapter(path: str | Path, *, role: str, iteration: int) -> str:
    completed = completed_adapter_path(path)
    if completed:
        return completed
    raise ACCCollabRegistryError(
        f"Missing completed {role} adapter for ACC-Collab iteration {iteration}: {path}"
    )


def actor_adapter_before(config: ACCCollabConfig, iteration: int) -> str | None:
    """Return Actor(i-1), or the configured iteration-zero Actor policy."""
    config.validate_iteration(iteration)
    if iteration == 1:
        initial = config.initialization.actor_adapter
        if initial:
            return _require_adapter(initial, role="initial actor", iteration=0)
        return None
    return _require_adapter(
        config.paths.adapter_dir(iteration - 1, "actor"),
        role="actor",
        iteration=iteration - 1,
    )


def critic_adapter_before(config: ACCCollabConfig, iteration: int) -> str | None:
    """Return Critic(i-1), or the configured iteration-zero Critic policy."""
    config.validate_iteration(iteration)
    if iteration == 1:
        initial = config.initialization.critic_adapter
        if initial:
            return _require_adapter(initial, role="initial critic", iteration=0)
        return None
    return _require_adapter(
        config.paths.adapter_dir(iteration - 1, "critic"),
        role="critic",
        iteration=iteration - 1,
    )


def critic_data_state(config: ACCCollabConfig, iteration: int) -> PolicyState:
    """Resolve the policies used to collect Critic preference data in iteration ``i``."""
    return PolicyState(
        actor_adapter=actor_adapter_before(config, iteration),
        critic_adapter=critic_adapter_before(config, iteration),
        actor_iteration=iteration - 1,
        critic_iteration=iteration - 1,
    )


def actor_data_state(config: ACCCollabConfig, iteration: int) -> PolicyState:
    """Resolve Actor(i-1) and the newly trained Critic(i)."""
    config.validate_iteration(iteration)
    return PolicyState(
        actor_adapter=actor_adapter_before(config, iteration),
        critic_adapter=_require_adapter(
            config.paths.adapter_dir(iteration, "critic"),
            role="critic",
            iteration=iteration,
        ),
        actor_iteration=iteration - 1,
        critic_iteration=iteration,
    )


def critic_training_initial_adapter(
    config: ACCCollabConfig,
    iteration: int,
) -> str | None:
    """Return the Critic policy from which Critic(i) must continue."""
    return critic_adapter_before(config, iteration)


def actor_training_initial_adapter(
    config: ACCCollabConfig,
    iteration: int,
) -> str | None:
    """Return the Actor policy from which Actor(i) must continue."""
    return actor_adapter_before(config, iteration)


def evaluation_state(config: ACCCollabConfig) -> PolicyState:
    """Resolve and authenticate the final Actor/Critic pair for evaluation.

    A normal training run evaluates adapters under its own ``run.output_dir``.
    Cross-dataset evaluation can instead set ``evaluation.policy_output_dir``;
    that mode treats the source run's ``registry/final.json`` as a signed-by-
    identity manifest and rejects stale, redirected, or mutated adapters.
    """
    source_output = config.evaluation.policy_output_dir
    if source_output is not None:
        return _external_evaluation_state(config, source_output)

    iteration = config.method.alternating_iterations
    return PolicyState(
        actor_adapter=_require_adapter(
            config.paths.adapter_dir(iteration, "actor"),
            role="actor",
            iteration=iteration,
        ),
        critic_adapter=_require_adapter(
            config.paths.adapter_dir(iteration, "critic"),
            role="critic",
            iteration=iteration,
        ),
        actor_iteration=iteration,
        critic_iteration=iteration,
    )


def _external_evaluation_state(
    config: ACCCollabConfig,
    source_output: str,
) -> PolicyState:
    source_paths = ACCCollabPaths(Path(source_output))
    registry_path = source_paths.final_registry
    try:
        payload = load_final_registry(registry_path)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise ACCCollabRegistryError(
            f"Cannot load external ACC-Collab final registry: {registry_path}"
        ) from exc

    iteration = config.method.alternating_iterations
    if payload.get("base_model") != config.model.name:
        raise ACCCollabRegistryError(
            "External registry base_model does not match evaluation model: "
            f"{payload.get('base_model')!r} != {config.model.name!r}"
        )
    if payload.get("alternating_iterations") != iteration:
        raise ACCCollabRegistryError(
            "External registry alternating_iterations does not match evaluation config: "
            f"{payload.get('alternating_iterations')!r} != {iteration}"
        )
    expected_method = "ACC-Collab+" if iteration == 2 else "ACC-Collab"
    if payload.get("method") != expected_method:
        raise ACCCollabRegistryError(
            f"External registry method must be {expected_method!r}, got {payload.get('method')!r}"
        )

    raw_state = payload.get("state")
    if not isinstance(raw_state, Mapping):
        raise ACCCollabRegistryError("External registry state must be a mapping")
    resolved: dict[str, str] = {}
    for role in ("actor", "critic"):
        iteration_key = f"{role}_iteration"
        if raw_state.get(iteration_key) != iteration:
            raise ACCCollabRegistryError(
                f"External registry {iteration_key} must be {iteration}, "
                f"got {raw_state.get(iteration_key)!r}"
            )
        adapter_key = f"{role}_adapter"
        policy_key = f"{role}_policy"
        registry_adapter = raw_state.get(adapter_key)
        if not isinstance(registry_adapter, str) or not registry_adapter:
            raise ACCCollabRegistryError(
                f"External registry {adapter_key} must name a completed adapter"
            )
        if raw_state.get(policy_key) != registry_adapter:
            raise ACCCollabRegistryError(
                f"External registry {policy_key} must equal {adapter_key}"
            )

        expected_adapter = _require_adapter(
            source_paths.adapter_dir(iteration, role),
            role=role,
            iteration=iteration,
        )
        registered_adapter = _require_adapter(
            registry_adapter,
            role=role,
            iteration=iteration,
        )
        if not _same_path(expected_adapter, registered_adapter):
            raise ACCCollabRegistryError(
                f"External registry redirects {role} outside the source run: "
                f"{registered_adapter} != {expected_adapter}"
            )
        actual_identity = path_identity(
            registered_adapter,
            hash_weights=bool(payload.get("identity_hash_weights", False)),
        )
        if payload.get(f"{role}_identity") != actual_identity:
            raise ACCCollabRegistryError(
                f"External registry {role}_identity does not match the completed adapter"
            )
        resolved[role] = registered_adapter

    return PolicyState(
        actor_adapter=resolved["actor"],
        critic_adapter=resolved["critic"],
        actor_iteration=iteration,
        critic_iteration=iteration,
    )


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve(strict=False) == Path(right).resolve(strict=False)


def build_iteration_registry(config: ACCCollabConfig, iteration: int) -> dict[str, Any]:
    """Build a completed iteration registry after Critic then Actor training."""
    config.validate_iteration(iteration)
    actor = _require_adapter(
        config.paths.adapter_dir(iteration, "actor"),
        role="actor",
        iteration=iteration,
    )
    critic = _require_adapter(
        config.paths.adapter_dir(iteration, "critic"),
        role="critic",
        iteration=iteration,
    )
    strict_identity = config.prompt_role.name != "original"
    payload = {
        "schema_version": 1,
        "pipeline": "acccollab_original",
        "status": "complete",
        "iteration": iteration,
        "base_model": config.model.name,
        "training_order": "critic_then_actor",
        "actor": {
            "adapter": actor,
            "initialized_from": actor_adapter_before(config, iteration) or "base_model",
            "identity": path_identity(actor, hash_weights=strict_identity),
        },
        "critic": {
            "adapter": critic,
            "initialized_from": critic_adapter_before(config, iteration) or "base_model",
            "identity": path_identity(critic, hash_weights=strict_identity),
        },
    }
    if strict_identity:
        payload["identity_hash_weights"] = True
    return payload


def write_iteration_registry(config: ACCCollabConfig, iteration: int) -> dict[str, Any]:
    """Atomically persist the registry for one finished alternation."""
    payload = build_iteration_registry(config, iteration)
    write_json(config.paths.iteration_registry(iteration), payload)
    return payload


def build_final_registry(config: ACCCollabConfig) -> dict[str, Any]:
    """Build the exact final Actor/Critic registry expected by evaluation."""
    state = evaluation_state(config)
    strict_identity = config.prompt_role.name != "original"
    payload = {
        "schema_version": 1,
        "pipeline": "acccollab_original",
        "status": "complete",
        "method": (
            "ACC-Collab+" if config.method.alternating_iterations == 2 else "ACC-Collab"
        ),
        "alternating_iterations": config.method.alternating_iterations,
        "base_model": config.model.name,
        "state": state.to_dict(),
        "actor_identity": path_identity(
            state.actor_adapter,
            hash_weights=strict_identity,
        ),
        "critic_identity": path_identity(
            state.critic_adapter,
            hash_weights=strict_identity,
        ),
    }
    if strict_identity:
        payload["identity_hash_weights"] = True
    return payload


def write_final_registry(config: ACCCollabConfig) -> dict[str, Any]:
    """Persist the final Actor/Critic policy pair used by evaluation."""
    payload = build_final_registry(config)
    write_json(config.paths.final_registry, payload)
    return payload


def iteration_registry_matches(config: ACCCollabConfig, iteration: int) -> bool:
    """Require an iteration registry to match current adapters and their identities exactly."""
    path = config.paths.iteration_registry(iteration)
    if not path.is_file():
        return False
    try:
        return read_json(path) == build_iteration_registry(config, iteration)
    except (ACCCollabRegistryError, OSError, TypeError, ValueError):
        return False


def final_registry_matches(config: ACCCollabConfig) -> bool:
    """Require the final registry to match the currently completed policy pair exactly."""
    path = config.paths.final_registry
    if not path.is_file():
        return False
    try:
        return read_json(path) == build_final_registry(config)
    except (ACCCollabRegistryError, OSError, TypeError, ValueError):
        return False


def load_final_registry(path: str | Path) -> dict[str, Any]:
    """Read and validate an ACC-Collab final registry."""
    payload = read_json(path)
    if (
        payload.get("schema_version") != 1
        or payload.get("pipeline") != "acccollab_original"
        or payload.get("status") != "complete"
    ):
        raise ACCCollabRegistryError(f"Invalid or incomplete ACC-Collab registry: {path}")
    return payload
