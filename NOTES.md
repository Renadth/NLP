# Lab Notes
## Lab 1 — Defect Safari
Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.
### Defect 1
- Class: Extra whitespace
- Example: FB-000008
- Why it matters: unnecessary whitespace increases wordcount 
- Decision: remove all extra spaces in preprocessing 
### Defect 2
- Class: HTML remnants
- Example: FB-000001
- Why it matters: leftover <br> is not natural language, may pollute the token stream
- Decision: clean the data from HTML remnants
### Defect 3
- Class: PII
- Example:  FB-000001
- Why it matters: Personal info must not get leaked into the training set
- Decision: mask all PII
### Defect 4
- Class: Repeated-character emphasis
- Example: FB-000088
- Why it matters: unbounded letter repetition causes sparse tokens, 
- Decision: Shorten letter repition but not fully because it indicates emphasis
### Defect 5
- Class: Duplicated words
- Example: FB-000015
- Why it matters: adds noise
- Decision: Remove duplicate
### Defect 6
- Class: Emoji
- Example: FB-000066
- Why it matters: It holds emphasis and meaning
- Decision: keep
## Lab 2 — Parameter audit
| Checkpoint | Total params | Embeddings % | Other notes |
|---|---:|---:|---|
| mBERT | | | |
| CAMeLBERT | | | |
## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:
‎README.md‎
