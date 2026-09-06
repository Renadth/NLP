# Lab Notes

## Lab 1 — Defect Safari
Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.

Defect 1

Class: Unicode form inconsistency
Example: "مـشكلة" (contains a presentation-form/extended Arabic character instead of standard form)
Why it matters: Same word appears as different byte sequences, so the model/tokenizer treats identical words as distinct tokens, hurting matching and training signal.
Decision: Apply Unicode NFKC normalization in normalize().

Defect 2

Class: Tatweel (kashida) elongation
Example: "الخدمــــة" (extra ـ characters stretching the word)
Why it matters: Tatweel is a purely visual/stylistic elongation with no semantic value; leaving it in inflates vocabulary and breaks tokenization consistency.
Decision: Strip all tatweel characters (\u0640) during normalization.

Defect 3

Class: Repeated-character emphasis (elongated spelling)
Example: "ممتااااز" (informal emphasis via repeated letters)
Why it matters: Users elongate words for emotional emphasis in informal text; unbounded repeats create sparse, noisy tokens the tokenizer has never seen.
Decision: Collapse runs of 3+ repeated characters down to 2, preserving the emphasis signal without unbounded noise.

Defect 4

Class: Code-switching (Arabic ↔ English)
Example: "الخدمة كانت bad جدا"
Why it matters: Mixed-language spans within a single sentence can confuse language-specific tokenizers and segmentation models if not handled explicitly.
Decision: Preserve mixed text as-is (no translation/removal); rely on a multilingual tokenizer to handle both scripts.

Defect 5

Class: Personally identifiable information (PII)
Example: "اتصل 0551234567" (phone number), "رقم الهوية 1023456789" (national ID)
Why it matters: Leaking PII into training data or logs is a privacy/compliance risk and adds spurious high-cardinality tokens.
Decision: Mask detected phone numbers and national-ID-shaped values with <PHONE> / <NATIONAL_ID> placeholders before training/eval.

Defect 6

Class: HTML remnants
Example: "الخدمة سيئة  <br>" (leftover HTML entities/tags from scraped or exported text)
Why it matters: Markup fragments are not natural language and add noise/tokens with no linguistic content.
Decision: Strip HTML tags and decode/remove HTML entities before normalization.

## Lab 2 — Parameter audit
| Checkpoint | Total params | Embeddings % | Other notes |
|---|---:|---:|---|
| mBERT | | | |
| CAMeLBERT | | | |

## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:
