# Error Taxonomy

# 

# Label ambiguity

# Arabic orthographic variation

# Dialect or code-switching

# Entity boundary or clitic alignment

# Long-context truncation

# Retrieval relevance mismatch

# Preprocessing or serving skew

# Annotation defect

# 

# Lab 6 hand-read result:

# \- Primary category: Label ambiguity

# \- Specific pattern: `parks → roads`

# \- Reviewed sample: 120 errors

\### Top 3 prioritised fixes



1\. Add hard-negative training examples distinguishing `parks` from `roads`.

2\. Increase Arabic training coverage for explicit park cues such as `حديقة`, playground, irrigation, and accessibility complaints.

3\. Add Arabic orthographic/noise variants for the parks class.



\### Predicted metric delta



The 300 observed validation errors represent 12.5% of the 2,400-row validation set. Therefore, correcting all 300 errors would have an upper-bound accuracy improvement of +12.5 percentage points. This is a theoretical ceiling, not a measured gain from the proposed fixes.

