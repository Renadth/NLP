# Decision Records

## tokenizer
- Chosen checkpoint(s): xlm-roberta-base (XLM-R)
- Arabic fertility evidence: 1.672 (lowest among multilingual candidates; mBERT=2.153, CAMeLBERT=1.405 is lower but specialized only for AR)
- English fertility evidence: 1.434 (best of the multilingual candidates; matches/beats mBERT's 1.510, far better than CAMeLBERT's 2.705)
- p95 length evidence: 19 tokens (AR), 21 tokens (EN) — the most balanced spread across languages of any candidate tested
- Operational trade-off / rationale: DistilBERT and CAMeLBERT each excel at one language but fail badly on the other (DistilBERT: AR fertility 4.527, p95=45 — effectively unusable for Arabic; CAMeLBERT: EN fertility 2.705, p95=36 — weak on English). Since Bayan serves both Arabic and English feedback in production, a single tokenizer that performs consistently on both languages is more valuable than one that excels on only one side at the other's expense. XLM-R is the only candidate meeting that bar.

## arabic-model
- Incumbent:
- Candidate:
- All/Gulf/MSA evidence:
- CI-backed verdict:
- Segmentation contract:

## search-min-score
- Threshold:
- No-answer evidence:
- False-positive / false-negative trade-off:

## quantisation-split
- Topic artefact:
- NER artefact:
- Latency evidence:
- Paired quality-tax evidence:
- Rollback artefact retained:

## architecture
- Encoder/decoder rationale by task:
- Multilingual vs Arabic-centric rationale:
- Evidence used:
