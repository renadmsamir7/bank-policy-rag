
import re


# Known merged-word errors observed in Qwen responses.
KNOWN_FORMATTING_ERRORS = {
    "whichthe": "which the",
    "withinthirty": "within thirty",
    "thisfee": "this fee",
}


def format_answer_text(answer):
    """
    Correct known merged-word errors in generated answers.

    Only explicitly recognized errors are corrected.
    Banking amounts, terminology, and other text are
    left unchanged.
    """

    pattern = re.compile(
        r"\b(" +
        "|".join(KNOWN_FORMATTING_ERRORS.keys()) +
        r")\b",
        flags=re.IGNORECASE
    )

    def replace_match(match):
        original = match.group(0)

        replacement = KNOWN_FORMATTING_ERRORS[
            original.casefold()
        ]

        # Preserve capitalization.
        if original.isupper():
            return replacement.upper()

        if original[0].isupper():
            return replacement.capitalize()

        return replacement

    return pattern.sub(replace_match, answer)
