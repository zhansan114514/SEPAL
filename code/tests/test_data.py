"""Tests for data loading, preprocessing, and sampling."""

import pytest
from unittest.mock import patch, MagicMock

from src.data.loader import load_dataset, validate_dataset_bundle


class TestDatasetLoader:
    """Test dataset loading functions (mocked for unit tests)."""

    @patch("src.data.loader.hf_load_dataset")
    def test_load_boolq_shape(self, mock_load):
        """BoolQ should return train/val/test splits."""
        mock_train = MagicMock()
        mock_train.__iter__ = MagicMock(return_value=iter([
            {"question": "Q?", "passage": "P", "answer": True},
        ]))
        mock_val = MagicMock()
        mock_val.__iter__ = MagicMock(return_value=iter([]))
        mock_ds = {"train": mock_train, "validation": mock_val}
        mock_load.return_value = mock_ds
        result = load_dataset("boolq")
        assert "train" in result
        assert len(result["train"]) == 1

    def test_load_unknown_dataset_raises(self):
        """Loading an unknown dataset should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset")

    @patch("src.data.mmlu.hf_load_dataset")
    @patch("src.data.mmlu.get_dataset_config_names")
    def test_load_mmlu_auxiliary_train_maps_to_train(self, mock_configs, mock_load):
        """MMLU auxiliary_train should be mapped to 'train' key."""
        mock_load.return_value = {
            "auxiliary_train": [
                {"question": "Q train", "choices": ["A", "B", "C", "D"], "answer": 1},
            ],
            "validation": [
                {"question": "Q val", "choices": ["A", "B", "C", "D"], "answer": 2},
            ],
            "test": [
                {"question": "Q test", "choices": ["A", "B", "C", "D"], "answer": 3},
            ],
            "dev": [
                {"question": "Q dev", "choices": ["A", "B", "C", "D"], "answer": 0},
            ],
        }
        mock_configs.return_value = []  # no per-subject loading needed for "all" mode

        from src.data.mmlu import load_mmlu
        data = load_mmlu(load_mode="all")

        assert "train" in data
        assert len(data["train"]) == 1
        assert data["train"][0]["answer"] == "B"
        assert data["train"][0]["source_split"] == "auxiliary_train"
        assert len(data["validation"]) == 1
        assert len(data["test"]) == 1
        assert len(data["dev"]) == 1

    @patch("src.data.mmlu.hf_load_dataset")
    @patch("src.data.mmlu.get_dataset_config_names")
    def test_load_mmlu_by_subject_preserves_subject(self, mock_configs, mock_load):
        """MMLU by_subject mode should preserve subject metadata."""
        # First call: load "all" config for auxiliary_train
        # Second+ calls: per-subject configs for val/test/dev
        all_data = {
            "auxiliary_train": [
                {"question": "Q train", "choices": ["A", "B"], "answer": 0},
            ],
            "validation": [],
            "test": [],
            "dev": [],
        }
        subject_data = {
            "validation": [{"question": "Q val", "choices": ["A", "B"], "answer": 1}],
            "test": [{"question": "Q test", "choices": ["A", "B"], "answer": 0}],
            "dev": [{"question": "Q dev", "choices": ["A", "B"], "answer": 1}],
        }

        mock_load.side_effect = [all_data, subject_data]
        mock_configs.return_value = ["abstract_algebra"]

        from src.data.mmlu import load_mmlu
        data = load_mmlu(load_mode="by_subject")

        assert data["test"][0]["subject"] == "abstract_algebra"
        assert data["validation"][0]["subject"] == "abstract_algebra"

    def test_validate_mmlu_empty_train_raises(self):
        """MMLU with empty train should raise ValueError."""
        data = {
            "train": [],
            "validation": [{"question": "v"}],
            "test": [{"question": "t"}],
        }
        with pytest.raises(ValueError, match="train.*empty"):
            validate_dataset_bundle("mmlu", data)

    def test_validate_mmlu_empty_test_raises(self):
        """MMLU with empty test should raise ValueError."""
        data = {
            "train": [{"question": "t"}],
            "validation": [{"question": "v"}],
            "test": [],
        }
        with pytest.raises(ValueError, match="test.*empty"):
            validate_dataset_bundle("mmlu", data)


class TestDataFormat:
    """Test that loaded data has consistent format."""

    def test_standardized_format_keys(self):
        """Each sample should have question, passage, answer, choices keys."""
        from src.data.preprocessor import standardize_sample

        # BoolQ sample
        sample = {
            "question": "Is the sky blue?",
            "passage": "The sky appears blue...",
            "answer": True,
        }
        result = standardize_sample(sample, task_type="yes_no")
        assert "question" in result
        assert "answer" in result

        # MMLU sample
        sample = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,  # index
        }
        result = standardize_sample(sample, task_type="multiple_choice")
        assert "question" in result
        assert "choices" in result
        assert "answer" in result

    def test_standardized_preserves_metadata(self):
        """Standardized samples should preserve dataset, source_split, subject."""
        from src.data.preprocessor import standardize_sample

        sample = {
            "dataset": "mmlu",
            "source_split": "test",
            "source_index": 42,
            "subject": "abstract_algebra",
            "category": "stem",
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,
        }
        result = standardize_sample(sample, task_type="multiple_choice")

        assert result["dataset"] == "mmlu"
        assert result["source_split"] == "test"
        assert result["source_index"] == 42
        assert result["subject"] == "abstract_algebra"
        assert result["category"] == "stem"
        assert result["raw_answer"] == "1"

    def test_standardized_metadata_defaults(self):
        """Missing metadata should get empty defaults."""
        from src.data.preprocessor import standardize_sample

        sample = {"question": "Q?", "answer": True}
        result = standardize_sample(sample, task_type="yes_no")

        assert result["dataset"] == ""
        assert result["source_split"] == ""
        assert result["source_index"] is None
        assert result["subject"] == ""
        assert result["category"] == ""


class TestDataPreprocessorEdgeCases:
    """Test edge cases and malformed inputs for data preprocessing."""

    def test_standardize_empty_sample(self):
        """Empty sample should return default structure."""
        from src.data.preprocessor import standardize_sample

        result = standardize_sample({}, task_type="yes_no")
        assert result["question"] == ""
        assert result["passage"] == ""
        assert result["answer"] == "no"  # Default for False/None

    def test_standardize_missing_fields(self):
        """Missing fields should use empty defaults."""
        from src.data.preprocessor import standardize_sample

        sample = {"answer": True}  # Missing question and passage
        result = standardize_sample(sample, task_type="yes_no")
        assert result["question"] == ""
        assert result["passage"] == ""
        assert result["answer"] == "yes"

    def test_standardize_boolq_varied_answer_formats(self):
        """Should handle different BoolQ answer formats."""
        from src.data.preprocessor import standardize_sample

        test_cases = [
            (True, "yes"),
            (False, "no"),
            ("true", "yes"),
            ("True", "yes"),
            ("false", "no"),
            ("False", "no"),
            (1, "no"),
            (0, "no"),
        ]
        for input_answer, expected in test_cases:
            sample = {"question": "Test?", "answer": input_answer}
            result = standardize_sample(sample, task_type="yes_no")
            assert result["answer"] == expected

    def test_standardize_mc_empty_choices(self):
        """Should handle empty choices list."""
        from src.data.preprocessor import standardize_sample

        sample = {
            "question": "Test?",
            "choices": [],
            "answer": "A",
        }
        result = standardize_sample(sample, task_type="multiple_choice")
        assert result["choices"] == []

    def test_standardize_mc_answer_as_index_out_of_bounds(self):
        """Answer index beyond choices length should return empty."""
        from src.data.preprocessor import standardize_sample

        sample = {
            "question": "Test?",
            "choices": ["A", "B"],
            "answer": 5,  # Out of bounds
        }
        result = standardize_sample(sample, task_type="multiple_choice")
        assert result["answer"] == ""

    def test_standardize_mc_answer_with_parentheses(self):
        """Should strip parentheses from answer string."""
        from src.data.preprocessor import standardize_sample

        sample = {
            "question": "Test?",
            "choices": ["A", "B", "C", "D"],
            "answer": "(B)",
        }
        result = standardize_sample(sample, task_type="multiple_choice")
        assert result["answer"] == "B"

    def test_standardize_mc_answer_lowercase(self):
        """Should uppercase lowercase answer letters."""
        from src.data.preprocessor import standardize_sample

        sample = {
            "question": "Test?",
            "choices": ["A", "B", "C", "D"],
            "answer": "c",
        }
        result = standardize_sample(sample, task_type="multiple_choice")
        assert result["answer"] == "C"

    def test_normalize_answer_with_none(self):
        """normalize_answer should handle None input."""
        from src.algorithms.reward import normalize_answer
        result = normalize_answer(None)
        assert result == ""

    def test_normalize_answer_with_empty_string(self):
        """Empty string should return empty."""
        from src.algorithms.reward import normalize_answer
        result = normalize_answer("")
        assert result == ""

    def test_normalize_answer_with_whitespace(self):
        """Should strip whitespace."""
        from src.algorithms.reward import normalize_answer
        result = normalize_answer("  YES  ")
        assert result == "Y"

    def test_extract_answer_with_whitespace_only(self):
        """Whitespace-only response should return None."""
        from src.algorithms.reward import extract_answer
        result = extract_answer("   \n\t  ", task_type="yes_no")
        assert result is None

    def test_extract_answer_with_very_long_text(self):
        """Should handle very long responses."""
        from src.algorithms.reward import extract_answer
        long_text = " ".join(["word"] * 10000) + " The answer is Yes."
        result = extract_answer(long_text, task_type="yes_no")
        assert result == "YES"

    def test_extract_answer_mixed_type_fallback(self):
        """Mixed type should fallback to yes_no when MC fails."""
        from src.algorithms.reward import extract_answer
        result = extract_answer("The answer is Yes.", task_type="mixed")
        assert result == "YES"

    def test_extract_answer_invalid_task_type(self):
        """Invalid task type should return None."""
        from src.algorithms.reward import extract_answer
        result = extract_answer("The answer is Yes.", task_type="invalid")
        assert result is None


class TestGenerateWrongAnswer:
    """Test wrong answer generation for guided trajectories."""

    def test_flip_yes_no(self):
        """Should flip yes to no and vice versa."""
        from src.data.preprocessor import generate_wrong_answer

        assert generate_wrong_answer("yes") == "no"
        assert generate_wrong_answer("no") == "yes"
        assert generate_wrong_answer("YES") == "no"
        assert generate_wrong_answer("No") == "yes"

    def test_pick_wrong_choice(self):
        """Should pick a different choice from the list."""
        from src.data.preprocessor import generate_wrong_answer

        import random
        with patch.object(random, "choice", side_effect=lambda x: x[0]):
            result = generate_wrong_answer("C", ["A", "B", "C", "D"])
            assert result != "C"
            assert result in ["A", "B", "D"]

    def test_empty_choices_uses_defaults(self):
        """Should use default A/B/C/D when choices is empty."""
        from src.data.preprocessor import generate_wrong_answer

        result = generate_wrong_answer("A", [])
        assert result in ["B", "C", "D"]

    def test_all_options_wrong(self):
        """When all options are 'wrong' (same as correct), should still return something."""
        from src.data.preprocessor import generate_wrong_answer

        result = generate_wrong_answer("A", ["A"])
        assert result in ["B", "C", "D"]

    def test_case_insensitive_comparison(self):
        """Should compare case-insensitively."""
        from src.data.preprocessor import generate_wrong_answer

        result = generate_wrong_answer("YES", ["yes", "no"])
        assert result == "no" or result == "NO"


class TestSampling:
    """Test the sampling module."""

    def test_full_sampling_returns_all(self):
        """Full strategy should return all samples."""
        from src.data.sampler import sample_split, SplitSamplingConfig

        samples = [{"id": i} for i in range(100)]
        result = sample_split(samples, SplitSamplingConfig(strategy="full"))
        assert len(result) == 100

    def test_full_with_none_max_samples(self):
        """None max_samples should return all."""
        from src.data.sampler import sample_split, SplitSamplingConfig

        samples = [{"id": i} for i in range(100)]
        result = sample_split(samples, SplitSamplingConfig(max_samples=None))
        assert len(result) == 100

    def test_random_sampling_not_prefix(self):
        """Random sampling should NOT return the first N items."""
        from src.data.sampler import sample_split, SplitSamplingConfig

        samples = [{"id": i} for i in range(100)]
        result = sample_split(
            samples,
            SplitSamplingConfig(strategy="random", max_samples=10, seed=42),
        )

        assert len(result) == 10
        # Very unlikely to be exactly [0,1,2,...,9] with random shuffle
        assert [x["id"] for x in result] != list(range(10))

    def test_random_sampling_reproducible(self):
        """Same seed should produce same result."""
        from src.data.sampler import sample_split, SplitSamplingConfig

        samples = [{"id": i} for i in range(100)]
        cfg = SplitSamplingConfig(strategy="random", max_samples=10, seed=42)

        r1 = sample_split(samples, cfg)
        r2 = sample_split(samples, cfg)
        assert r1 == r2

    def test_random_different_seeds_different_results(self):
        """Different seeds should (likely) produce different results."""
        from src.data.sampler import sample_split, SplitSamplingConfig

        samples = [{"id": i} for i in range(100)]
        r1 = sample_split(samples, SplitSamplingConfig(strategy="random", max_samples=20, seed=42))
        r2 = sample_split(samples, SplitSamplingConfig(strategy="random", max_samples=20, seed=99))

        # Very unlikely to be identical
        ids1 = [x["id"] for x in r1]
        ids2 = [x["id"] for x in r2]
        assert ids1 != ids2

    def test_stratified_sampling_covers_all_subjects(self):
        """Stratified sampling should cover all subjects."""
        from src.data.sampler import sample_split, SplitSamplingConfig

        samples = []
        for subject in ["math", "history", "law"]:
            for i in range(10):
                samples.append({"subject": subject, "id": f"{subject}-{i}"})

        result = sample_split(
            samples,
            SplitSamplingConfig(
                strategy="stratified_by_subject",
                max_samples=6,
                seed=42,
                group_key="subject",
            ),
        )

        assert len(result) == 6
        subjects = {x["subject"] for x in result}
        assert subjects == {"math", "history", "law"}

    def test_stratified_with_many_groups(self):
        """Stratified sampling with more groups than samples."""
        from src.data.sampler import sample_split, SplitSamplingConfig

        samples = []
        for subject in [f"sub_{i}" for i in range(20)]:
            samples.append({"subject": subject, "id": subject})

        result = sample_split(
            samples,
            SplitSamplingConfig(
                strategy="stratified_by_subject",
                max_samples=10,
                seed=42,
                group_key="subject",
            ),
        )

        assert len(result) == 10
        # Should cover at least 10 different subjects
        subjects = {x["subject"] for x in result}
        assert len(subjects) >= 10

    def test_sampling_small_dataset(self):
        """If dataset is smaller than max_samples, return all."""
        from src.data.sampler import sample_split, SplitSamplingConfig

        samples = [{"id": i} for i in range(5)]
        result = sample_split(
            samples,
            SplitSamplingConfig(strategy="random", max_samples=100, seed=42),
        )
        assert len(result) == 5

    def test_sampling_empty_dataset(self):
        """Empty dataset should return empty list."""
        from src.data.sampler import sample_split, SplitSamplingConfig

        result = sample_split(
            [],
            SplitSamplingConfig(strategy="random", max_samples=10, seed=42),
        )
        assert result == []

    def test_invalid_max_samples_raises(self):
        """Negative max_samples should raise ValueError."""
        from src.data.sampler import sample_split, SplitSamplingConfig

        with pytest.raises(ValueError, match="max_samples must be positive"):
            sample_split(
                [{"id": 1}],
                SplitSamplingConfig(strategy="random", max_samples=-1, seed=42),
            )

    def test_unknown_strategy_raises(self):
        """Unknown strategy should raise ValueError."""
        from src.data.sampler import sample_split, SplitSamplingConfig

        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            sample_split(
                [{"id": 1}, {"id": 2}],
                SplitSamplingConfig(strategy="nonexistent", max_samples=1, seed=42),
            )

    def test_apply_sampling(self):
        """apply_sampling should apply per-split config."""
        from src.data.sampler import apply_sampling

        data = {
            "train": [{"id": i} for i in range(100)],
            "test": [{"id": i} for i in range(50)],
        }

        result = apply_sampling(
            data,
            {
                "train": {"strategy": "random", "max_samples": 10, "seed_offset": 0},
                "test": {"strategy": "full", "max_samples": None, "seed_offset": 2},
            },
            base_seed=42,
        )

        assert len(result["train"]) == 10
        assert len(result["test"]) == 50

    def test_apply_sampling_no_config_returns_all(self):
        """apply_sampling with no config for a split should return all."""
        from src.data.sampler import apply_sampling

        data = {
            "train": [{"id": i} for i in range(50)],
            "test": [{"id": i} for i in range(25)],
        }

        result = apply_sampling(data, {}, base_seed=42)
        assert len(result["train"]) == 50
        assert len(result["test"]) == 25


class TestBBHDatasetLoading:
    """Test BBH dataset loading with custom splitting."""

    @patch("src.data.loader.hf_load_dataset")
    def test_bbh_custom_split_ratios(self, mock_hf_load):
        """BBH should accept custom split ratios."""
        from src.data.loader import load_dataset
        from datasets import DatasetDict, Dataset

        task1_data = Dataset.from_dict({
            "question": [f"Task1 Q{i}" for i in range(50)],
            "answer": ["A"] * 50,
        })
        task2_data = Dataset.from_dict({
            "question": [f"Task2 Q{i}" for i in range(50)],
            "answer": ["B"] * 50,
        })

        mock_dataset_dict = DatasetDict({
            "task1": task1_data,
            "task2": task2_data,
        })
        mock_hf_load.return_value = mock_dataset_dict

        result = load_dataset(
            "bbh",
            split_ratios={"test": 0.3, "validation": 0.2},
            seed=42,
        )

        assert "train" in result
        assert "validation" in result
        assert "test" in result

    @patch("src.data.loader.hf_load_dataset")
    def test_bbh_default_split_ratios(self, mock_hf_load):
        """BBH should use default split ratios when none provided."""
        from src.data.loader import load_dataset
        from datasets import DatasetDict, Dataset

        task_data = Dataset.from_dict({
            "question": [f"Q{i}" for i in range(100)],
            "answer": ["A"] * 100,
        })

        mock_dataset_dict = DatasetDict({
            "task1": task_data,
        })
        mock_hf_load.return_value = mock_dataset_dict

        result = load_dataset("bbh")

        assert "train" in result
        assert "validation" in result
        assert "test" in result

    @patch("src.data.loader.hf_load_dataset")
    def test_bbh_stratified_by_task(self, mock_hf_load):
        """BBH samples should preserve task information."""
        from src.data.loader import load_dataset
        from datasets import DatasetDict, Dataset

        task1_data = Dataset.from_dict({
            "question": [f"task1 Q{i}" for i in range(50)],
            "answer": ["A"] * 50,
        })
        task2_data = Dataset.from_dict({
            "question": [f"task2 Q{i}" for i in range(50)],
            "answer": ["B"] * 50,
        })

        mock_dataset_dict = DatasetDict({
            "task1": task1_data,
            "task2": task2_data,
        })
        mock_hf_load.return_value = mock_dataset_dict

        result = load_dataset("bbh", seed=42)

        all_samples = (
            result.get("train", []) +
            result.get("validation", []) +
            result.get("test", [])
        )
        assert len(all_samples) == 100

    @patch("src.data.loader.hf_load_dataset")
    def test_bbh_empty_dataset(self, mock_hf_load):
        """BBH with empty dataset should raise ValueError (validation catches it)."""
        from src.data.loader import load_dataset
        from datasets import DatasetDict, Dataset

        empty_data = Dataset.from_dict({
            "question": [],
            "answer": [],
        })

        mock_dataset_dict = DatasetDict({
            "task1": empty_data,
        })
        mock_hf_load.return_value = mock_dataset_dict

        # Empty dataset should now raise because train split is empty
        with pytest.raises(ValueError, match="train.*empty"):
            load_dataset("bbh")

    @patch("src.data.loader.hf_load_dataset")
    def test_bbh_single_task(self, mock_hf_load):
        """BBH with single task should still work."""
        from src.data.loader import load_dataset
        from datasets import DatasetDict, Dataset

        task_data = Dataset.from_dict({
            "question": [f"Q{i}" for i in range(100)],
            "answer": ["A"] * 100,
        })

        mock_dataset_dict = DatasetDict({
            "only_task": task_data,
        })
        mock_hf_load.return_value = mock_dataset_dict

        result = load_dataset("bbh")

        total = len(result.get("train", [])) + len(result.get("validation", [])) + len(result.get("test", []))
        assert total == 100

    def test_paper_bbh_split_is_exact_stratified_and_task_typed(self):
        from src.data.loader import (
            PAPER_BBH_MULTIPLE_CHOICE_TASKS,
            PAPER_BBH_TASKS,
            PAPER_BBH_YES_NO_TASKS,
            _load_bbh,
        )

        raw = {}
        for task_name in PAPER_BBH_TASKS:
            if task_name in PAPER_BBH_MULTIPLE_CHOICE_TASKS:
                raw[task_name] = [
                    {
                        "input": (
                            f"Question {index}?\nOptions:\n(A) zero\n(B) one\n"
                            "(C) two\n(D) three\n(E) four\n(F) five\n(G) six"
                        ),
                        "target": "(F)",
                    }
                    for index in range(100)
                ]
            else:
                raw[task_name] = [
                    {
                        "input": (
                            "True and False is"
                            if task_name == "boolean_expressions"
                            else f"Question {index}?"
                        ),
                        "target": "False" if task_name == "boolean_expressions" else "No",
                    }
                    for index in range(100)
                ]

        result = _load_bbh(raw, "mixed", split_ratios=None, seed=101)
        assert len(result["test"]) == 1260
        assert len(result["validation"]) == 500
        assert len(result["train"]) == 440
        assert {sample["subject"] for sample in result["test"]} == set(PAPER_BBH_TASKS)
        assert {sample["subject"] for sample in result["validation"]} == set(PAPER_BBH_TASKS)

        mc = next(sample for sample in result["test"] if sample["subject"] == "temporal_sequences")
        assert mc["task_type"] == "multiple_choice"
        assert mc["answer"] == "F"
        assert mc["choice_labels"] == list("ABCDEFG")
        assert "Options:" not in mc["question"]

        yn = next(sample for sample in result["test"] if sample["subject"] in PAPER_BBH_YES_NO_TASKS)
        assert yn["task_type"] == "yes_no"
        assert yn["answer"] == "no"

        repeated = _load_bbh(raw, "mixed", split_ratios=None, seed=101)
        identity = lambda sample: (sample["subject"], sample["source_index"])
        assert [identity(sample) for sample in result["test"]] == [
            identity(sample) for sample in repeated["test"]
        ]

    def test_arc_numeric_source_label_maps_to_canonical_letter(self):
        from src.data.preprocessor import standardize_sample

        result = standardize_sample(
            {
                "question": "Pick the third option",
                "choices": {"text": ["x", "y", "z", "w"], "label": ["1", "2", "3", "4"]},
                "answerKey": "3",
            },
            task_type="multiple_choice",
        )
        assert result["choice_labels"] == ["A", "B", "C", "D"]
        assert result["source_choice_labels"] == ["1", "2", "3", "4"]
        assert result["answer"] == "C"

    def test_malformed_bbh_free_text_target_is_explicitly_unscorable(self):
        from src.data.loader import _standardize_bbh_sample

        result = _standardize_bbh_sample(
            {
                "bbh_task": "movie_recommendation",
                "input": "Pick one.\nOptions:\n(A) Alpha\n(B) Beta",
                "target": "Monsters, Inc",
                "source_index": 17,
            },
            fallback_task_type="mixed",
        )

        assert result["answer"] == "__UNSCORABLE__"
        assert result["bbh_raw_target"] == "Monsters, Inc"
        assert result["bbh_source_index"] == 17

    def test_truncated_bbh_single_option_record_remains_reproducible(self):
        from src.data.loader import _standardize_bbh_sample

        result = _standardize_bbh_sample(
            {
                "bbh_task": "snarks",
                "input": "Which statement is sarcastic?\nOptions:\n(A) The NB",
                "target": "(A)",
                "source_index": 88,
            },
            fallback_task_type="mixed",
        )

        assert result["choices"] == ["The NB"]
        assert result["answer"] == "A"
        assert result["bbh_source_warning"] == "fewer_than_two_options"
