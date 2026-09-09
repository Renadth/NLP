"""Lab 4: per-model Arabic normalisation profiles."""

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    normalized = text

    # Remove tatweel / kashida.
    normalized = normalized.replace("ـ", "")

    # Remove Arabic diacritics when requested.
    if profile.dediacritize:
        normalized = "".join(
            ch
            for ch in normalized
            if unicodedata.category(ch) != "Mn"
        )

    if profile.name == "bayan_ar_v1":
        # Normalize common Arabic letter variants.
        normalized = normalized.translate(
            str.maketrans(
                {
                    "أ": "ا",
                    "إ": "ا",
                    "آ": "ا",
                    "ٱ": "ا",
                    "ى": "ي",
                    "ة": "ه",
                    "ؤ": "و",
                }
            )
        )
    else:
        raise ValueError(
            f"Unknown Arabic normalization profile: {profile.name}"
        )

    # Normalize whitespace.
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def segment(text: str) -> list[str]:
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tokenizers.morphological import MorphologicalTokenizer
    from camel_tools.tokenizers.word import simple_word_tokenize

    words = simple_word_tokenize(text)

    disambiguator = MLEDisambiguator.pretrained("calima-msa-r13")

    tokenizer = MorphologicalTokenizer(
        disambiguator=disambiguator,
        scheme="atbtok",
        split=True,
        diac=False,
    )

    return tokenizer.tokenize(words)