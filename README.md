# CNLPModels.jl

[![CI](https://github.com/madsuite-org/CNLPModels.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/madsuite-org/CNLPModels.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/madsuite-org/CNLPModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/madsuite-org/CNLPModels.jl)

Load a nonlinear program (NLP) exposed by a **shared library through a plain
C interface** and use it as an `NLPModels.AbstractNLPModel` — so any
NLPModels-compatible solver (Ipopt via NLPModelsIpopt, or any other) can
solve it in Julia.

Any library implementing the ABI below works, in any source language:
`test/fixtures/tinyqp.c` in this repository is a complete reference
implementation in ~100 lines of plain C.

```julia
using CNLPModels, NLPModelsIpopt

CNLPModels.set_path!("/opt/models")        # or the CNLPMODELS_PATH env variable

m = CNLPModel("@mymodel", 1000)            # resolves libmymodel.so, instance of size 1000
# m = CNLPModel("/path/to/mymodel", 1000)  # or a literal path: library file or bundle dir
res = ipopt(m)
```

The `cnlp"..."` literal is the shortcut onto the path — it resolves the
name, checks that the library exists and loads, and hands back the cached
handle, directly usable as the library argument:

```julia
m = CNLPModel(cnlp"mymodel", 1000)
```

(Or bypass the path entirely with `CNLPModels.load("/path/libmymodel.so")`.)
Any number of model instances may coexist per library.

## The C ABI

For a chosen symbol prefix `P` (default `"rec"`; the name-based loader
defaults it to the library name), the library exports the functions below.
Conventions throughout:

- every function returns `int32` **status**: `0` = success (except `P_new` /
  `P_data_begin` / `P_new_from_data` / `P_schema`, which return positive
  values on success as described);
- indices are **1-based**;
- the Hessian is the **lower triangle** of `obj_weight * ∇²f(x) + Σᵢ yᵢ ∇²cᵢ(x)`;
- arrays are dense `double`/`int32`/`int64` buffers allocated by the caller.

### Instantiation

```c
int32_t P_new(int32_t n);
```
Create a model instance of size `n`. **Returns a positive model id**, `0` on
failure. Instances live for the process lifetime; ids are never reused.
(For libraries whose instantiation needs structured data rather than one
integer, see the builder ABI below.)

### Metadata (per instance `id`)

```c
int32_t P_nvar(int32_t id);    // number of variables
int32_t P_ncon(int32_t id);    // number of constraints
int32_t P_nnzj(int32_t id);    // Jacobian nonzeros
int32_t P_nnzh(int32_t id);    // Hessian nonzeros (lower triangle)
int32_t P_meta(int32_t id, double* x0, double* lvar, double* uvar,
               double* lcon, double* ucon);
```
`P_meta` fills the initial point, variable bounds (length `nvar`) and
constraint bounds (length `ncon`); use `±INFINITY` for unbounded.

### Evaluation (per instance `id`; `x` has length `nvar`)

```c
int32_t P_obj (int32_t id, const double* x, double* out);          // f(x) into *out
int32_t P_grad(int32_t id, const double* x, double* g);            // ∇f(x), length nvar
int32_t P_cons(int32_t id, const double* x, double* c);            // c(x), length ncon
int32_t P_jac_structure(int32_t id, int32_t* rows, int32_t* cols); // length nnzj, 1-based
int32_t P_jac (int32_t id, const double* x, double* vals);         // length nnzj
int32_t P_hess_structure(int32_t id, int32_t* rows, int32_t* cols);// length nnzh, 1-based
int32_t P_hess(int32_t id, const double* x, const double* y,       // y has length ncon
               double obj_weight, double* vals);                   // length nnzh
```

### Structured instantiation (optional, "ABI v2")

Libraries whose models are built from structured data (tables, arrays,
scalars) publish a schema and take the data through a builder:

```c
int32_t P_schema(uint8_t* buf, int32_t len);   // returns needed length; fills buf
```
The schema is JSON:
`{"abi":2,"fields":[{"name":"...","kind":"scalar"|"array"|"table",
"type":"f64"|"i64", "columns":[{"name":"...","type":"..."}]}]}`.

```c
int32_t P_data_begin(void);                                        // → builder id (>0)
int32_t P_set_scalar_f64(int32_t b, const char* field, double v);  // and _i64 (int64_t)
int32_t P_set_array_f64 (int32_t b, const char* field, const double* v, int32_t len);
int32_t P_set_col_f64   (int32_t b, const char* table, const char* col,
                         const double* v, int32_t len);            // and _i64 variants
int32_t P_data_ready    (int32_t b);   // 1 iff every slot is filled consistently
int32_t P_new_from_data (int32_t b);   // → model id (>0), 0 on failure
```
Tables cross the boundary **as columns** of equal length; the library
reassembles rows internally. From Julia this is transparent: the arguments are
positional, one per schema field, in the order the library publishes them —

```julia
m = CNLPModel("@mymodel",
    [(i = 1, pd = 0.4), (i = 2, pd = 0.3)],   # a table
    [0.9, 0.9],                               # an array
    100.0,                                    # a scalar
)
m = CNLPModel("@scalable", 1000)              # one integer: <prefix>_new when
                                              # exported, else the builder
```

which is the same spelling the producer side uses — `ExaModel(core, arg1,
arg2, ...)` to instantiate a recipe, `compile_library(out, core, arg1, ...)`
to compile one — so a model is consumed the way it was written. A library
compiled from a recipe names its fields `arg1`, `arg2`, ... for that reason.

## Implementing a compatible library

1. Pick a prefix and export the instantiation, metadata, and evaluation
   functions above with C linkage.
2. Keep instances behind integer ids (a static table suffices — see the
   reference implementation).
3. Return `0`/positive ids on success, nonzero/`0` on failure — never throw
   across the boundary.
4. Check yourself against `test/fixtures/tinyqp.c` and this package's test
   suite, which compiles that file and exercises every function, including
   the closed-form solution of the model it implements. The file carries two
   models: `tq_`, instantiated from one integer, and `sq_`, which has no
   one-integer constructor and is built from a three-field schema through the
   builder.

## Notes for libraries carrying their own runtime

If the shared library embeds a language runtime that initializes lazily on
first call and re-forwards the process-global BLAS trampoline (symptom:
`Error: no BLAS/LAPACK library loaded for dgemm_()` in the host afterwards),
this package repairs the host's BLAS forwarding automatically after the
library's first call; `restore_blas!(lib)` re-applies it at any time.

Sibling package: [`cnlpmodels`](https://github.com/madsuite-org/cnlpmodels-py) —
the same consumer for Python (ctypes + numpy).
