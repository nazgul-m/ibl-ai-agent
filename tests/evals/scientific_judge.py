
"""Evidence-grounded scientific reasoning judge using Claude CLI.

Scores answers from 0 to 2; normalized score is score / 2.
Importing this module does not invoke Claude.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Callable

from tests.evals.scientific_judge_prompt import build_judge_prompt


@dataclass(frozen=True)
class Judgment:
    """Scientific evaluation result.

    Attributes
    ----------
    score : int
        Score from 0 to 2.
    reason : str
        Explanation of the score.
    unsupported_claims : list[str]
        Claims not supported by evidence.
    """

    score: int
    reason: str
    unsupported_claims: list[str]

    @property
    def normalized_score(self) -> float:
        """Normalize the score.

        Returns
        -------
        float
            Score divided by 2.
        """
        return self.score / 2.0


def parse_judgment(raw: str) -> Judgment:
    """Validate and parse a judge response.

    Parameters
    ----------
    raw : str
        JSON response from the judge.

    Returns
    -------
    Judgment
        Validated scientific judgment.
    """
    try:
        obj = json.loads(raw.strip())
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Judge did not return valid JSON") from exc

    if (
        not isinstance(obj, dict)
        or set(obj) != {"score", "reason", "unsupported_claims"}
    ):
        raise ValueError(
            "Judge JSON must have score, reason, unsupported_claims"
        )

    if type(obj["score"]) is not int or obj["score"] not in (0, 1, 2):
        raise ValueError("Judge score must be integer 0, 1, or 2")

    if not isinstance(obj["reason"], str) or not obj["reason"].strip():
        raise ValueError("Judge reason must be nonempty text")

    if (
        not isinstance(obj["unsupported_claims"], list)
        or not all(
            isinstance(x, str) and bool(x.strip())
            for x in obj["unsupported_claims"]
        )
    ):
        raise ValueError(
            "unsupported_claims must be a list of strings"
        )

    return Judgment(
        score=obj["score"],
        reason=obj["reason"],
        unsupported_claims=obj["unsupported_claims"],
    )


def claude_cli_generate(
    prompt: str,
    model: str = "sonnet",
    timeout: int = 120,
) -> str:
    """Call Claude CLI with tools disabled.

    Parameters
    ----------
    prompt : str
        Evaluation prompt.
    model : str
        Claude model alias.
    timeout : int
        Timeout in seconds.

    Returns
    -------
    str
        CLI standard output.
    """
    result = subprocess.run(
        [
            "claude",
            "-p",
            prompt,
            "--model",
            model,
            "--tools",
            "",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )

    return result.stdout


def judge_answer(
    question: str,
    answer: str,
    evidence: str,
    generate: Callable[[str], str] | None = None,
) -> Judgment:
    """Score an answer, retrying once for malformed output.

    Parameters
    ----------
    question, answer, evidence : str
        Question, agent answer, and available evidence.
    generate : Callable or None
        Optional generator for mocked tests.

    Returns
    -------
    Judgment
        Validated judgment.

    Notes
    -----
    Backend errors propagate. Valid scores are never retried.
    """
    prompt = build_judge_prompt(question, answer, evidence)
    generator = generate or claude_cli_generate

    for attempt in range(2):
        response = generator(prompt)

        try:
            return parse_judgment(response)

        except ValueError as exc:
            if attempt == 1:
                raise ValueError(
                    "Judge returned an invalid response after 2 attempts. "
                    f"Last response preview: {str(response)[:200]!r}"
                ) from exc

    raise RuntimeError("Unexpected judge retry state")
