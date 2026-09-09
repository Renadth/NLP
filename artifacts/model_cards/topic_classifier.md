# Bayan Topic Classifier

## Model
CAMeL-Lab/bert-base-arabic-camelbert-mix

## Task
Eight-class bilingual municipal topic classification.

## Evidence
Validation macro-F1: 1.0000
Frozen test macro-F1: 1.0000
Validation error pool: 300 / 2400
Dominant observed confusion: parks -> roads

## Known limitations
Evaluation data are synthetic and may overstate generalisation.
The validation error taxonomy shows a strong parks/roads confusion pattern.
These results should not be interpreted as production-grade accuracy on naturally occurring citizen language.
