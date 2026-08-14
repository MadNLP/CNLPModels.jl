/* cnlp.h — the cnlp ABI: nonlinear programs behind shared libraries.
 *
 * Version 0.1 (2026-08-14).  THIS HEADER IS THE SPECIFICATION — there is no
 * separate prose document to drift from it.  The README records how it
 * relates to CUTEst, CppADCodeGen, CasADi, FMI, ASL and NLPModels.jl.
 * Worked code: example/qp.c + example/solve.c in this repository (one
 * model, both sides, ~170 lines); the conformance tests are the consumer
 * suites (CNLPModels.jl, cnlpmodels-py), whose fixture `tinyqp.c` is a
 * fuller multi-model reference.
 *
 * MODEL.  A library carries any number of MODELS, each under a symbol
 * prefix that IS its name.  A model is instantiated into INSTANCES behind
 * positive integer ids; instances are evaluated through sparse first- and
 * second-order callbacks.  Two library-level symbols have fixed names;
 * everything else is per model.  Because symbols are per-prefix, this
 * header declares them through X-macros:
 *
 *     #include "cnlp.h"
 *     CNLP_DECLARE_MODEL(acopf)         // required: core + instantiation
 *         // "##" below is C token pasting: this expands to acopf_new,
 *         // acopf_obj, ... — the exported symbols carry no macro
 *         // machinery.  See example/qp.c for a complete worked model.
 *     CNLP_DECLARE_NEW_STR(acopf)       // if the entry point is a string
 *     CNLP_DECLARE_BUILDER(acopf)       // if the model is structured
 *     CNLP_DECLARE_LAYOUT(acopf)        // if it publishes named blocks
 *
 * SEMANTICS.  The evaluation surface is a C transliteration of
 * NLPModels.jl's API (https://jso.dev/NLPModels.jl/stable/api/), inheriting
 * that ecosystem's conventions unchanged:
 *
 *   - indices are 1-BASED inside the ABI (Ipopt vocabulary: index_style =
 *     FORTRAN_STYLE); consumers shift to their language's convention;
 *   - sparse structures are COO triplets;
 *   - the Hessian is the LOWER TRIANGLE of
 *         obj_weight * grad^2 f(x)  +  sum_i y_i * grad^2 c_i(x);
 *   - buffers are caller-allocated dense double / int32 / int64 arrays.
 *
 * STATUSES.  Every function returns int32.  0 = success, except the
 * functions documented as returning a positive id, count, or byte length —
 * for those, 0 from an id-returning function means instantiation failure,
 * and -1 means an index out of range.  1 = invalid id, unknown index, or a
 * block of the wrong kind; 3 = a length disagreeing with the addressed
 * object; any other nonzero from an evaluator is a model-defined failure.
 * Never let an exception or longjmp cross the boundary.
 *
 * STRINGS use one copy-out convention everywhere: the function returns the
 * needed byte length and copies what fits in `cap` — call once with
 * (NULL, 0) to size, once to fill.  UTF-8, not NUL-terminated within the
 * returned length.
 *
 * TIERS.  A library implements an optional tier fully or not at all.
 * Consumers probe ONE witness symbol per tier (noted at each block below)
 * and degrade gracefully: a tier's absence is an answer, never an error.
 *
 * SCOPE.  The ABI publishes only what a library of ANY origin can honestly
 * answer: names, sizes, offsets, dimensions, argument types, descriptions.
 * It deliberately stops short of a model's algebraic structure — an
 * expression tree is a particular modelling package's representation, and
 * asking for one would make this that package's interface in C clothing
 * rather than a generic abstraction over "an NLP behind a C boundary".
 *
 * CANDIDATE EXTENSIONS (absent today, in likely order): Hessian/Jacobian-
 * vector products (P_hprod / P_jprod — consumers currently assemble
 * products from the COO callbacks); a library-level version symbol
 * (cnlp_abi_version, CppADCodeGen's pattern) once the packages register;
 * constraint linearity/equality classification and integer variables;
 * precision/index-width variants (CUTEst's suffix discipline is the
 * template); an explicit reentrancy contract (today: distinct instances
 * from distinct threads is expected to work, a single instance is not
 * promised reentrant); per-block descriptions (P_block_desc).
 */
#ifndef CNLP_H
#define CNLP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Library level (optional; witness: cnlp_nmodels) ──────────────────────
 * What the library carries.  The only fixed-name symbols in the ABI: every
 * per-model entry point needs a prefix, and this is where a prefix comes
 * from.  Model k's name (k in 0..n-1) is returned by the string convention
 * above; the name IS the symbol prefix its functions are exported under. */
int32_t cnlp_nmodels(void);
int32_t cnlp_model_name(int32_t k, uint8_t* buf, int32_t cap);

/* ── Core + instantiation, per prefix P (required) ────────────────────────
 * Exactly ONE of P##_new / P##_new_str / the builder is the model's entry
 * point.  P##_argtype says which, and what to pass, WITHOUT symbol probing:
 * it publishes what the model instantiates FROM as a comma-separated list
 * of types, each optionally `type|description`:
 *
 *     ""                                          fixed; P##_new, arg ignored
 *     "int|size"                                  P##_new(n)
 *     "string|path to a case file"                P##_new_str(s)
 *     "int|arg1,Vector{f64}|v0,Table{i::int w::f64}|bus"    the builder,
 *                                                 one value per schema field
 *
 * P##_nargs is the count of values that close the model (0 = fixed). */
#define CNLP_DECLARE_MODEL(P)                                                 \
    int32_t P##_nargs(void);                                                  \
    int32_t P##_argtype(uint8_t* buf, int32_t cap);                           \
    int32_t P##_new(int32_t n);                  /* -> id > 0; 0 = failure */ \
    /* metadata, per instance id.  P##_meta fills caller buffers of length    \
       nvar (x0, lvar, uvar) and ncon (lcon, ucon); +-INFINITY is fine. */    \
    int32_t P##_nvar(int32_t id);                                             \
    int32_t P##_ncon(int32_t id);                                             \
    int32_t P##_nnzj(int32_t id);                                             \
    int32_t P##_nnzh(int32_t id);                                             \
    int32_t P##_meta(int32_t id, double* x0, double* lvar, double* uvar,      \
                     double* lcon, double* ucon);                             \
    /* evaluation, per instance id; buffer lengths in comments */             \
    int32_t P##_obj (int32_t id, const double* x, double* out);  /* 1     */ \
    int32_t P##_grad(int32_t id, const double* x, double* g);    /* nvar  */ \
    int32_t P##_cons(int32_t id, const double* x, double* c);    /* ncon  */ \
    int32_t P##_jac_structure(int32_t id, int32_t* rows, int32_t* cols);      \
    int32_t P##_jac (int32_t id, const double* x, double* vals); /* nnzj  */ \
    int32_t P##_hess_structure(int32_t id, int32_t* rows, int32_t* cols);     \
    int32_t P##_hess(int32_t id, const double* x, const double* y,            \
                     double obj_weight, double* vals);           /* nnzh  */

/* String entry point — a model instantiated from one string (a case-file
 * path, say).  Declared separately because most models do not have it. */
#define CNLP_DECLARE_NEW_STR(P)                                               \
    int32_t P##_new_str(const char* s);          /* -> id > 0; 0 = failure */

/* ── Structured instantiation (optional per model; witness: P##_schema) ───
 * For models built from structured data.  P##_schema returns JSON:
 *
 *   {"fields":[{"name":...,"kind":"scalar"|"array"|"table",
 *               "type":"i64"|"f64", ...}, ...]}
 *
 * (tables carry "columns").  The consumer opens a builder, sets one value
 * per field BY NAME — tables cross the boundary as columns — and asks for
 * an instance.  Values are checked against the declared kind and type;
 * nothing is coerced.  P##_data_ready returns 1 iff the builder's data is
 * complete and consistent; the library decides, not the consumer. */
#define CNLP_DECLARE_BUILDER(P)                                               \
    int32_t P##_schema(uint8_t* buf, int32_t cap);                            \
    int32_t P##_data_begin(void);                /* -> builder id > 0      */ \
    int32_t P##_set_scalar_i64(int32_t b, const char* field, int64_t v);      \
    int32_t P##_set_scalar_f64(int32_t b, const char* field, double v);       \
    int32_t P##_set_array_i64 (int32_t b, const char* field,                  \
                               const int64_t* v, int32_t len);                \
    int32_t P##_set_array_f64 (int32_t b, const char* field,                  \
                               const double* v, int32_t len);                 \
    int32_t P##_set_col_i64   (int32_t b, const char* table, const char* col, \
                               const int64_t* v, int32_t len);                \
    int32_t P##_set_col_f64   (int32_t b, const char* table, const char* col, \
                               const double* v, int32_t len);                 \
    int32_t P##_data_ready    (int32_t b);       /* 1 iff complete         */ \
    int32_t P##_new_from_data (int32_t b);       /* -> id > 0; 0 = failure */

/* ── Named blocks (optional per model; witness: P##_nblocks) ──────────────
 * The blocks the model's author NAMED — variables, constraints, parameters
 * — so a consumer holding only the library addresses solution slices by
 * name.  P##_block fills out = [kind, offset, length, ndims, dims...] with
 * kind 0 = variable, 1 = constraint, 2 = parameter and offset 0-BASED into
 * the corresponding vector (the solution, the constraint/multiplier vector,
 * or the parameter vector).  The layout is INSTANCE state: a model
 * instantiated at another size reports that size's offsets.  Unnamed blocks
 * still occupy the vectors — offsets account for them — but are not
 * published: the layout answers "what did this model name", not "what does
 * the solution vector contain".
 *
 * Parameters are LIVE, PER-INSTANCE state: P##_set_value takes effect for
 * every later evaluation of that instance, and other instances keep their
 * own values.  get/set on a non-parameter block returns 1; a length
 * disagreeing with the block returns 3. */
#define CNLP_DECLARE_LAYOUT(P)                                                \
    int32_t P##_nblocks(int32_t id);                                          \
    int32_t P##_block(int32_t id, int32_t k, int32_t* out);                   \
    int32_t P##_block_name(int32_t id, int32_t k, uint8_t* buf, int32_t cap); \
    int32_t P##_get_value(int32_t id, int32_t k, double* vals, int32_t len);  \
    int32_t P##_set_value(int32_t id, int32_t k,                              \
                          const double* vals, int32_t len);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CNLP_H */
