# NWP abliation

The idea is to see the effect of the different NWP variables

Let's od this two ways.
1. Remove the variable and see the difference
2. Only train with one variable

Let's start by looking at 8 hour horizon

| Variable | Remove | Only  |
|----------|--------|-------|
| dswrf    | 0.027  | 0.028 |
| lcc      | 0.026  | 0.029 |
| t        | 0.026  | 0.037 |
| wdir10   | 0.028  | 0.036 |
