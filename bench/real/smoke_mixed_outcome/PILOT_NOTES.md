# streaming_watermark_dedupe pilot - n=120
Cost: $1.6807  (0.0140 / trace)
Fixtures: {'streaming_watermark_dedupe': 120}

## 2x2 contingency (public_pass x hidden_pass)
|             | hidden=T | hidden=F |
|-------------|---------:|---------:|
| **public=T**|       56 |       64 |
| **public=F**|        0 |        0 |

## Decision gate (task 6.3)
Generalization-gap cell `(public=T, hidden=F)`: 64 (need >=3)
Gate: **PASS**

## Failure modes
| slice | count |
|-------|------:|
| exception | 0 |
| format_failure | 0 |
| hidden_semantic_failure | 64 |
| pass | 56 |
| public_failure | 0 |
| timeout | 0 |

## Failure modes by fixture
| slice | count |
|-------|------:|
| fixture=streaming_watermark_dedupe,mode=hidden_semantic_failure | 64 |
| fixture=streaming_watermark_dedupe,mode=pass | 56 |

## Failure modes by model
| slice | count |
|-------|------:|
| model=large,mode=hidden_semantic_failure | 15 |
| model=large,mode=pass | 50 |
| model=small,mode=hidden_semantic_failure | 49 |
| model=small,mode=pass | 6 |

## Showcase composition gate
Requires hidden-semantic failures in both model arms and format failures not to be the majority of failures.
Gate: **PASS**

## Extraction failures
No fenced-code extraction failures.
