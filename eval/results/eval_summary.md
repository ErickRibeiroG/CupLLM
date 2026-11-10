### Summary

| subset | model | n | fact coverage (%) | fully correct (%) | mean tokens |
|---|---|---|---|---|---|
| Cuphead (domínio) | base | 29 | 24.7 | 13.8 | 373 |
| Cuphead (domínio) | finetuned | 29 | 25.9 | 20.7 | 129 |
| Fora do domínio | base | 5 | 90.0 | 80.0 | 42 |
| Fora do domínio | finetuned | 5 | 80.0 | 80.0 | 57 |

### By training coverage (in-domain)

| training_coverage | model | n | fact_coverage | complete | mean_tokens |
|---|---|---|---|---|---|
| coberto | base | 14 | 25.0 | 21.4 | 367 |
| coberto | finetuned | 14 | 28.6 | 21.4 | 111 |
| novo | base | 10 | 26.7 | 10.0 | 321 |
| novo | finetuned | 10 | 35.0 | 30.0 | 123 |
| parcial | base | 5 | 20.0 | 0.0 | 496 |
| parcial | finetuned | 5 | 0.0 | 0.0 | 191 |

Paired comparison (in-domain, n=29): fine-tuned better on 5, tie on 20, worse on 4.