# DeterminantalPointProcesses

[![Build Badge](https://travis-ci.org/alshedivat/DeterminantalPointProcesses.jl.svg)](https://travis-ci.org/alshedivat/DeterminantalPointProcesses.jl)
[![Coverage Badge](https://coveralls.io/repos/github/alshedivat/DeterminantalPointProcesses.jl/badge.svg)](https://coveralls.io/github/alshedivat/DeterminantalPointProcesses.jl)

An efficient implementation of Determinantal Point Processes (DPP) in Julia.

### Current features
- Exact sampling [1] from DPP and k-DPP (can be executed in parallel).
- MCMC sampling [2] from DPP and k-DPP (parallelization will be added).
- `pmf` and `logpmf` evaluation functions [1] for DPP and k-DPP.

### Planned features
- Exact sampling using dual representation [1].
- Better integration with MCMC frameworks in Julia (such as [Lora.jl]).
- Fitting DPP and k-DPP models to data [3, 4].
- Reduced rank DPP and k-DPP.
- Kronecker Determinantal Point Processes [5].

### Contributing
Currently, no timeline, no milestones, no promisses.

Contributions are sought (especially if you are an author of a related paper).
Bug reports are welcome.

## References
[1] Kulesza, A., and B. Taskar. Determinantal point processes for machine learning. [arXiv:1207.6083], 2012.

[2] Kang, B. Fast determinantal point process sampling with application to clustering. NIPS, 2013.

[3] Gillenwater, J., A. Kulesza, E. Fox, and B. Taskar. Expectation-Maximization for learning Determinantal Point Processes. NIPS, 2014.

[4] Mariet, Z., and S. Sra. Fixed-point algorithms for learning determinantal point processes. NIPS, 2015.

[5] Mariet, Z., and S. Sra. Kronecker Determinantal Point Processes. [arXiv:1605.08374], 2016.


[Julia-0.4 Badge]: http://pkg.julialang.org/badges/DeterminantalPointProcesses_0.4.svg
[Julia-0.5 Badge]: http://pkg.julialang.org/badges/DeterminantalPointProcesses_0.5.svg
[DPP-pkg]: http://pkg.julialang.org/?pkg=DeterminantalPointProcesses

[Build Badge]: https://travis-ci.org/alshedivat/DeterminantalPointProcesses.jl.svg
[Build Status]: https://travis-ci.org/alshedivat/DeterminantalPointProcesses.jl
[Coverage Badge]: https://coveralls.io/repos/github/alshedivat/DeterminantalPointProcesses.jl/badge.svg
[Coverage Status]: https://coveralls.io/github/alshedivat/DeterminantalPointProcesses.jl

[Lora.jl]: https://github.com/JuliaStats/Lora.jl
[arXiv:1207.6083]: https://arxiv.org/abs/1207.6083
[arXiv:1605.08374]: https://arxiv.org/abs/1605.08374
