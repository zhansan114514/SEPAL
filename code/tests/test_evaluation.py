"""Tests for answer extraction and accuracy evaluation."""

import pytest

from src.algorithms.reward import (
    extract_answer,
    extract_answer_with_source,
    normalize_answer,
)
from src.evaluation.answer_resolution import (
    answers_match,
    compute_accuracy_mixed,
)
from src.paired.generation import resolve_final_decision


class TestResolveFinalDecision:
    """Majority-then-judge final-answer policy for 3-Actor deliberation."""

    def test_majority_when_two_agree(self):
        source, final, majority, count = resolve_final_decision(
            ["A", "A", "B"], "multiple_choice"
        )
        assert source == "majority"
        assert final == "A"
        assert majority == "A"
        assert count == 2

    def test_majority_when_all_three_agree(self):
        source, final, _, count = resolve_final_decision(
            ["C", "C", "C"], "multiple_choice"
        )
        assert source == "majority"
        assert final == "C"
        assert count == 3

    def test_judge_when_all_three_differ(self):
        source, final, majority, count = resolve_final_decision(
            ["A", "B", "C"], "multiple_choice"
        )
        assert source == "judge"
        assert final is None  # awaiting the judge
        assert count == 1

    def test_judge_when_answers_missing(self):
        source, final, _, count = resolve_final_decision(
            [None, None, None], "multiple_choice"
        )
        assert source == "judge"
        assert final is None
        assert count == 0

    def test_two_unparseable_falls_back_to_judge(self):
        # Only one parseable answer -> no majority of >=2.
        source, final, _, count = resolve_final_decision(
            ["A", None, None], "multiple_choice"
        )
        assert source == "judge"
        assert final is None
        assert count == 1


class TestExtractAnswer:
    """Test the zeta function for answer extraction."""

    # --- Yes/No ---
    @pytest.mark.parametrize(
        "response, expected",
        [
            ("The answer is Yes.", "YES"),
            ("Final answer: No", "NO"),
            ("Based on the passage, the answer is Yes", "YES"),
            ("I think the answer is no.", "NO"),
            ("Yes, that is correct.", "YES"),
            ("No, that is incorrect.", "NO"),
            ("YES", "YES"),
            ("NO", "NO"),
        ],
    )
    def test_yes_no(self, response, expected):
        result = extract_answer(response, task_type="yes_no")
        assert result == expected

    def test_yes_no_empty(self):
        assert extract_answer("", task_type="yes_no") is None

    def test_yes_no_no_keyword(self):
        """Response without yes/no should return None."""
        assert extract_answer("The sky is blue.", task_type="yes_no") is None

    @pytest.mark.parametrize(
        "response, expected_source",
        [
            ("**Final Answer:** Yes\n\nJustification: supported.", "final_answer"),
            ("**Answer:** Yes\n\n**Justification:** supported.", "tail_claim"),
            (
                "Based on the passage, I conclude that:\n\n**Yes**\n\n"
                "Justification: supported.",
                "weak_tail",
            ),
        ],
    )
    def test_yes_no_markdown_anchors_from_boolq_smoke(
        self, response, expected_source
    ):
        result = extract_answer_with_source(response, task_type="yes_no")
        assert result.answer == "YES"
        assert result.source == expected_source

    @pytest.mark.parametrize(
        "response, expected",
        [
            (
                "Yes. The passage explicitly supports the statement.",
                "YES",
            ),
            (
                "No. The passage contradicts the statement.",
                "NO",
            ),
            ("Yes! This follows directly from the passage.", "YES"),
            ("No — the passage describes the opposite.", "NO"),
        ],
    )
    def test_yes_no_leading_answer_with_same_line_explanation(
        self, response, expected
    ):
        result = extract_answer_with_source(response, task_type="yes_no")
        assert result.answer == expected
        assert result.source == "weak_tail"

    @pytest.mark.parametrize(
        "response",
        [
            "No matter which option we consider, the passage is ambiguous.",
            "Yes or no questions require a binary answer.",
        ],
    )
    def test_yes_no_leading_word_without_answer_punctuation_is_not_extracted(
        self, response
    ):
        result = extract_answer_with_source(response, task_type="yes_no")
        assert result.answer is None
        assert result.source == "none"

    # --- Multiple Choice ---
    @pytest.mark.parametrize(
        "response, expected",
        [
            ("The answer is (A).", "A"),
            ("Final Answer: (B)", "B"),
            ("I choose C", "C"),
            ("The correct option is D.", "D"),
            ("Answer: A", "A"),
            ("(C) is the right answer.", "C"),
            ("Option B is correct.", "B"),
            # "final ANSWER is" (the "final RESULT is" anchor does not cover this).
            ("After reasoning, the final answer is (B) because.", "B"),
            ("The final answer is:\n\n(A) some option.", "A"),
            # "best/correct/final/likely ... is" + colon/newline/ellipsis.
            ("the best answer is (D) 51.59.", "D"),
            ("The correct answer is:\n\n(B) incomplete dominance", "B"),
            ("I conclude the correct answer is... (D) value.", "D"),
            # "choose" family incl. apostrophe contractions and "going to/with".
            ("I'd choose (B) Our interpretation.", "B"),
            ("I'll choose:\n\n(A) paralysis.", "A"),
            ("I'm going to choose:\n\n(D) 200 mounds.", "D"),
            ("I'm going with (A) that option.", "A"),
            ("I would choose:\n\n(B) Asia.", "B"),
            # "my selection/answer would be" and "diagnosis is".
            ("My selection would be (C).", "C"),
            ("the most likely diagnosis is:\n\n(C) Graves disease.", "C"),
            # Trailing "(X)." weak fallback.
            ("William Penn was a member of the Quakers (C).", "C"),
        ],
    )
    def test_multiple_choice(self, response, expected):
        result = extract_answer(response, task_type="multiple_choice")
        assert result == expected

    def test_mc_empty(self):
        assert extract_answer("", task_type="multiple_choice") is None

    def test_mc_markdown_anchor(self):
        result = extract_answer_with_source(
            "**Final Answer:** B\n\nJustification: supported.",
            task_type="multiple_choice",
        )
        assert result.answer == "B"
        assert result.source == "final_answer"

    # --- Mixed ---
    def test_mixed_prefers_mc(self):
        result = extract_answer("The answer is (A).", task_type="mixed")
        assert result == "A"

    def test_mixed_fallback_yes_no(self):
        result = extract_answer("The answer is Yes.", task_type="mixed")
        assert result == "YES"

    def test_final_result_extraction_source(self):
        result = extract_answer_with_source("The final result is B.", task_type="multiple_choice")
        assert result.answer == "B"
        assert result.source == "final_result"

    def test_think_block_is_ignored_before_answer_extraction(self):
        result = extract_answer_with_source(
            "<think>The final result is A.</think>\nReasoning.\nThe final result is C.",
            task_type="multiple_choice",
        )
        assert result.answer == "C"
        assert result.source == "final_result"

    def test_stray_think_close_tag_is_ignored_before_answer_extraction(self):
        result = extract_answer_with_source(
            "Reasoning.\nThe final result is A.\n</think>\nReasoning.\nThe final result is C.",
            task_type="multiple_choice",
        )
        assert result.answer == "C"
        assert result.source == "final_result"

    def test_stray_think_close_tag_does_not_drop_visible_answer(self):
        result = extract_answer_with_source(
            "Reasoning.\nThe final result is A.\n</think>",
            task_type="multiple_choice",
        )
        assert result.answer == "A"
        assert result.source == "final_result"

    def test_unterminated_think_block_does_not_produce_answer(self):
        result = extract_answer_with_source(
            "<think>The answer is A and then the output was truncated",
            task_type="multiple_choice",
        )
        assert result.answer is None
        assert result.source == "none"

    def test_mid_response_unterminated_think_block_is_ignored(self):
        result = extract_answer_with_source(
            "Visible prefix.\n<think>The final result is A and generation stopped",
            task_type="multiple_choice",
        )
        assert result.answer is None
        assert result.source == "none"

    def test_tail_claim_extraction_source(self):
        result = extract_answer_with_source(
            "Therefore, the answer is B.",
            task_type="multiple_choice",
        )
        assert result.answer == "B"
        assert result.source == "tail_claim"

class TestNormalizeAnswer:
    def test_yes(self):
        assert normalize_answer("Yes") == "Y"

    def test_no(self):
        assert normalize_answer("No") == "N"

    def test_option_a(self):
        assert normalize_answer("(A)") == "A"

    def test_option_b(self):
        assert normalize_answer("B.") == "B"

    def test_empty(self):
        assert normalize_answer("") == ""


class TestMixedTaskAccuracy:
    def test_mixed_task_types_accuracy(self):
        preds = ["A", "YES", "42"]
        labels = ["A", "Yes", "42.0"]
        task_types = ["multiple_choice", "yes_no", "math"]
        assert compute_accuracy_mixed(preds, labels, task_types) == 1.0


def test_bbh_extended_multiple_choice_labels_are_extracted_and_matched():
    result = extract_answer("Reasoning. Final Answer: (G)", task_type="multiple_choice")
    assert result == "G"
    assert answers_match(result, "G", "multiple_choice")


def test_extended_parser_does_not_treat_ordinary_word_as_choice():
    result = extract_answer("The answer is wrong.", task_type="multiple_choice")
    assert result is None


class TestExtractAnswerEdgeCases:
    """Test edge cases and malformed inputs for answer extraction."""

    def test_whitespace_only_response(self):
        """Response with only whitespace should return None."""
        assert extract_answer("   \n\t  ", task_type="yes_no") is None
        assert extract_answer("   \n\t  ", task_type="multiple_choice") is None

    def test_mixed_case_yes_no(self):
        """Extract should work with various case combinations."""
        test_cases = ["yEs", "yeS", "YEs", "yES", "YeS", "yEs"]
        for case in test_cases:
            result = extract_answer(f"The answer is {case}.", task_type="yes_no")
            # Should normalize to YES/NO
            assert result in ("YES", "NO") or result is None

    def test_multiple_yes_no_uses_last(self):
        """When multiple Yes/No appear, should extract the last one."""
        response = "Yes it could be, but actually No in the end."
        result = extract_answer(response, task_type="yes_no")
        assert result == "NO"

    def test_yes_no_with_punctuation(self):
        """Yes/No with various punctuation should still extract."""
        test_cases = [
            ("Yes!", "YES"),
            ("No?", "NO"),
            ("Yes.", "YES"),
            ("No,", "NO"),
            ("Yes:", "YES"),
        ]
        for response, expected in test_cases:
            result = extract_answer(f"Final answer: {response}", task_type="yes_no")
            assert result == expected

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("Based on the passage, my answer is: No.", "NO"),
            ('The answer to the question "Is healthcare free?" is No.', "NO"),
            ("My answer to the yes-no question is: Yes", "YES"),
            ("The final answer is:\n\nB: No\n\nJustification follows.", "NO"),
            ("Final Answer: B: No\n\nJustification follows.", "NO"),
            ("Final Answer: A: Yes\n\nJustification follows.", "YES"),
        ],
    )
    def test_yes_no_accepts_explicit_answer_with_incidental_option_label(
        self,
        response,
        expected,
    ):
        """Peer debate may prefix an explicit Yes/No with an A/B label."""
        assert extract_answer(response, task_type="yes_no") == expected

    @pytest.mark.parametrize(
        ("response", "expected", "expected_source"),
        [
            ("Final Answer: B) No", "NO", "final_answer"),
            ("Answer: A) Yes", "YES", "tail_claim"),
            ("A: Yes", "YES", "tail_claim"),
            ("B (No)", "NO", "tail_claim"),
            ("After considering the peers, I conclude: B: No", "NO", "tail_claim"),
            ("**Final Answer:** B", "NO", "final_answer"),
            ("**Answer:** A\n\nJustification follows.", "YES", "tail_claim"),
            ("Reasoning concludes here.\n(B)", "NO", "weak_tail"),
            (
                "B: The passage states that the two objects are identical.",
                "NO",
                "weak_tail",
            ),
            ("The final result is A.", "YES", "final_result"),
        ],
    )
    def test_yes_no_accepts_binary_choice_convention_from_peer_debate(
        self,
        response,
        expected,
        expected_source,
    ):
        result = extract_answer_with_source(response, task_type="yes_no")
        assert result.answer == expected
        assert result.source == expected_source

    def test_yes_no_explicit_word_overrides_conflicting_binary_label(self):
        result = extract_answer_with_source(
            "Several peers chose B.\nFinal Answer: A: No",
            task_type="yes_no",
        )
        assert result.answer == "NO"
        assert result.source == "final_answer"

    @pytest.mark.parametrize(
        "response",
        [
            "Person A disagrees with Person B.",
            "The answer begins with a broad discussion.",
        ],
    )
    def test_yes_no_does_not_treat_unanchored_binary_letters_as_answers(
        self,
        response,
    ):
        result = extract_answer_with_source(response, task_type="yes_no")
        assert result.answer is None
        assert result.source == "none"

    def test_multiple_choice_with_punctuation(self):
        """Multiple choice with various punctuation."""
        test_cases = [
            ("The answer is (A).", "A"),
            ("The answer is (B)!", "B"),
            ("The answer is (C)?", "C"),
            ("The answer is (D),", "D"),
        ]
        for response, expected in test_cases:
            result = extract_answer(response, task_type="multiple_choice")
            assert result == expected

    def test_very_long_response(self):
        """Should handle very long responses without crashing."""
        long_text = " ".join(["word"] * 10000) + " The answer is Yes."
        result = extract_answer(long_text, task_type="yes_no")
        assert result == "YES"

    def test_response_with_special_chars(self):
        """Special characters should not break extraction."""
        special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        response = f"Final answer: Yes {special_chars}"
        result = extract_answer(response, task_type="yes_no")
        assert result == "YES"

    def test_unicode_characters(self):
        """Should handle unicode characters gracefully."""
        response = "The answer is Yes 🎉 你好"
        result = extract_answer(response, task_type="yes_no")
        assert result == "YES"

    def test_mixed_task_type_with_only_mc(self):
        """Mixed task type should prefer MC when only MC present."""
        response = "The answer is (A)."
        result = extract_answer(response, task_type="mixed")
        assert result == "A"

    def test_mixed_task_type_with_only_yn(self):
        """Mixed task type should fallback to yes_no when no MC present."""
        response = "The answer is Yes."
        result = extract_answer(response, task_type="mixed")
        assert result == "YES"

    def test_invalid_task_type(self):
        """Invalid task type should return None and log warning."""
        result = extract_answer("The answer is Yes.", task_type="invalid_type")
        assert result is None

    def test_newline_in_answer(self):
        """Answer split across newlines should still be extracted."""
        response = "Final\nanswer:\nYes"
        result = extract_answer(response, task_type="yes_no")
        # May not extract due to pattern matching, but should not crash
        assert result is None or result in ("YES", "NO")

    def test_bbh_extended_option_range_extracts_only_through_r(self):
        """BBH may contain up to 18 choices, labelled A through R."""
        assert extract_answer("The answer is (E).", task_type="multiple_choice") == "E"
        assert extract_answer("Final Answer: R", task_type="multiple_choice") == "R"
        assert extract_answer("Final Answer: S", task_type="multiple_choice") is None

    def test_nested_parentheses(self):
        """Nested parentheses should still extract correctly."""
        result = extract_answer("The answer is ((A)).", task_type="multiple_choice")
        assert result == "A"
