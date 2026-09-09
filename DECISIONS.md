# Decision Records

## tokenizer

* Chosen checkpoint(s): xlm-roberta-base (XLM-R)
* Arabic fertility evidence: 1.672 (lowest among multilingual candidates; mBERT=2.153, CAMeLBERT=1.405 is lower but specialized only for AR)
* English fertility evidence: 1.434 (best of the multilingual candidates; matches/beats mBERT's 1.510, far better than CAMeLBERT's 2.705)
* p95 length evidence: 19 tokens (AR), 21 tokens (EN) — the most balanced spread across languages of any candidate tested
* Operational trade-off / rationale: DistilBERT and CAMeLBERT each excel at one language but fail badly on the other (DistilBERT: AR fertility 4.527, p95=45 — effectively unusable for Arabic; CAMeLBERT: EN fertility 2.705, p95=36 — weak on English). Since Bayan serves both Arabic and English feedback in production, a single tokenizer that performs consistently on both languages is more valuable than one that excels on only one side at the other's expense. XLM-R is the only candidate meeting that bar.

## arabic-model

Incumbent:

CAMeLBERT-mix (`CAMeL-Lab/bert-base-arabic-camelbert-mix`)



Candidate:

CAMeLBERT-DA (`CAMeL-Lab/bert-base-arabic-camelbert-da`)



All/Gulf/MSA evidence:

\- CAMeLBERT-mix: All 1.0000 / Gulf 1.0000 / MSA 1.0000

\- CAMeLBERT-DA: All 1.0000 / Gulf 1.0000 / MSA 1.0000

\- Gulf macro-F1 delta (DA − mix): +0.0000

\- Target: +0.0400

\- Result: target not met



CI-backed verdict:

Not computed; the bake-off reported point estimates only. Based on the measured slice results, there is no observed Gulf-slice advantage for CAMeLBERT-DA.



Segmentation contract:

Step 3 completed. LOCATION recall was 1.00 for both the original and segmented NER runs, giving a delta of +0.00 percentage points; the approximately +4-point target was not met on the supplied dataset.search-min-score

* Threshold:
* No-answer evidence:
* False-positive / false-negative trade-off:



## quantisation-split

* Topic artefact:
* NER artefact:
* Latency evidence:
* Paired quality-tax evidence:
* Rollback artefact retained:

## architecture

* Encoder/decoder rationale by task:
* Multilingual vs Arabic-centric rationale:
* Evidence used:

