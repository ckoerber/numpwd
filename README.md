# numpwd

## Description

### Idea / Workflow

1. Do a spin decomposition of given operator into two-nucleon channels
2. Compute all allowed channels for specified j, mj, l and operator transitions
3. Do analytical pwd in big Phi for allowed mla, s, ms, sp, msp transitions (improvement: lazy + cached, async if on GPU)
4. Do numeric integration for all unique non-zero analytical pwds and all remaining two-n channels:
    1. Group-by same (l s)j mj combinations (in/out)
    2. Apply sum and xi xo integration (improvement: apply sum in right order such that cache is small and cleared once no remaining dependency)
5. Export result to HDF 5
    1. array of shape (q, channel, po, pi) for given input ranges (include op meta (sympy))
    2. channel map (include channel meta for op)
    3. q values
    4. pi, po, wi, wo values (include mesh meta)
    5. Integration grids (include op mesh meta)

## Install
Install via pip
```bash
pip install [-e] [--user] .
```

## Run


## Authors
* {author}

## Contribute

## License
See [LICENSE.md](LICENSE.md)
