# CNLPModels.jl

Consume an NLP exposed by a shared library through a plain C interface as an
`NLPModels.AbstractNLPModel`, so any NLPModels-compatible solver (MadNLP,
Ipopt, ...) can solve it in Julia.

The main producer of such libraries is ExaModels' recorder
(`record`/`replay`) AOT-compiled with `juliac --output-lib --trim=safe`
(bundled and privatized), but any library implementing the ABI works, in any
language — `test/fixtures/tinyqp.c` is a complete reference implementation in
~100 lines of C.

```julia
using CNLPModels, MadNLP

lib = CNLPModels.load("librecorder.so")          # snapshots host BLAS, dlopens
m = CNLPModel(lib; prefix = "rec", n = 1000)     # rec_new(1000) → id, BLAS restore
res = madnlp(m)
```

Two things this package handles for you:

1. **The NLPModels wrapping** — meta, sparsity structures, and the seven
   evaluation callbacks, forwarded through cached function pointers with
   status checking and counters.
2. **The libblastrampoline seam** — a `juliac`-compiled library carries its
   own (privatized) Julia runtime, but libblastrampoline is process-global,
   and the library's lazy runtime initialization silently clears the host's
   BLAS forwarding (symptom: `no BLAS/LAPACK library loaded for dgemm_()`).
   `load` snapshots the host's BLAS configuration before `dlopen`, and the
   `CNLPModel` constructor restores it after the library's first call.
   `restore_blas!(lib)` can be called again at any time.

The C ABI convention (1-based indices, lower-triangle Lagrangian Hessian with
`obj_weight`, `Cint` status returns) is documented in the module docstring
(`?CNLPModels`) and mirrored by the C fixture.
