/*
 * Minimal reference implementation of the CNLPModels C ABI, in plain C —
 * both a hermetic test fixture and documentation-by-example that the ABI is
 * language-neutral.
 *
 *   min  sum_i (x_i - 1)^2   s.t.  x_1 + x_2 = 1        (n >= 2 variables)
 *
 * Optimum: x_1 = x_2 = 1/2, x_i = 1 (i >= 3), objective 1/2.
 * Conventions: 1-based indices, lower-triangle Hessian of
 * obj_weight * f + sum_i y_i c_i (the constraint is linear, so y drops out).
 */
#include <stdint.h>
#include <math.h>

#define TQ_MAX_MODELS 16
static int32_t Ns[TQ_MAX_MODELS];
static int32_t n_models = 0;

/* Returns a positive model id, or 0 on failure. */
int32_t tq_new(int32_t n) {
    if (n < 2 || n_models >= TQ_MAX_MODELS) return 0;
    Ns[n_models++] = n;
    return n_models;
}

/* ── ABI v2: schema + builder ─────────────────────────────────────────────
 * The same model exposed through the structured-data surface, so consumers'
 * builder paths (named tuples, positional tuples) are testable against this
 * fixture too. One scalar field: n. */
static const char tq_schema_str[] =
    "{\"abi\":2,\"fields\":[{\"name\":\"n\",\"kind\":\"scalar\",\"type\":\"i64\"}]}";

/* Fills buf (up to cap bytes) and returns the schema's full length. */
int32_t tq_schema(char *buf, int32_t cap) {
    int32_t len = (int32_t)(sizeof(tq_schema_str) - 1);
    for (int32_t i = 0; i < cap && i < len; i++) buf[i] = tq_schema_str[i];
    return len;
}

#define TQ_MAX_BUILDERS 16
static int64_t builder_n[TQ_MAX_BUILDERS];
static int32_t builder_set[TQ_MAX_BUILDERS];
static int32_t n_builders = 0;

int32_t tq_data_begin(void) {
    if (n_builders >= TQ_MAX_BUILDERS) return 0;
    builder_n[n_builders] = 0;
    builder_set[n_builders] = 0;
    n_builders++;
    return n_builders;
}

static int strsame(const char *a, const char *b) {
    while (*a && *b && *a == *b) { a++; b++; }
    return *a == 0 && *b == 0;
}

int32_t tq_set_scalar_i64(int32_t b, const char *field, int64_t v) {
    if (b < 1 || b > n_builders || !strsame(field, "n")) return 1;
    builder_n[b - 1] = v;
    builder_set[b - 1] = 1;
    return 0;
}

/* 1 = complete and consistent, 0 = not. */
int32_t tq_data_ready(int32_t b) {
    return (b >= 1 && b <= n_builders && builder_set[b - 1]) ? 1 : 0;
}

int32_t tq_new_from_data(int32_t b) {
    if (tq_data_ready(b) != 1) return 0;
    return tq_new((int32_t)builder_n[b - 1]);
}

static int32_t getN(int32_t id) {
    return (id >= 1 && id <= n_models) ? Ns[id - 1] : -1;
}

int32_t tq_nvar(int32_t id) { return getN(id); }
int32_t tq_ncon(int32_t id) { return getN(id) > 0 ? 1 : -1; }
int32_t tq_nnzj(int32_t id) { return getN(id) > 0 ? 2 : -1; }
int32_t tq_nnzh(int32_t id) { return getN(id); }

int32_t tq_meta(int32_t id, double *x0, double *lvar, double *uvar, double *lcon, double *ucon) {
    int32_t N = getN(id);
    if (N < 0) return 1;
    for (int32_t i = 0; i < N; i++) {
        x0[i] = 0.0;
        lvar[i] = -INFINITY;
        uvar[i] = INFINITY;
    }
    lcon[0] = 0.0;
    ucon[0] = 0.0;
    return 0;
}

int32_t tq_obj(int32_t id, const double *x, double *out) {
    int32_t N = getN(id);
    if (N < 0) return 1;
    double s = 0.0;
    for (int32_t i = 0; i < N; i++) {
        double d = x[i] - 1.0;
        s += d * d;
    }
    *out = s;
    return 0;
}

int32_t tq_grad(int32_t id, const double *x, double *g) {
    int32_t N = getN(id);
    if (N < 0) return 1;
    for (int32_t i = 0; i < N; i++) g[i] = 2.0 * (x[i] - 1.0);
    return 0;
}

int32_t tq_cons(int32_t id, const double *x, double *c) {
    c[0] = x[0] + x[1] - 1.0;
    return 0;
}

int32_t tq_jac_structure(int32_t id, int32_t *rows, int32_t *cols) {
    rows[0] = 1; cols[0] = 1;
    rows[1] = 1; cols[1] = 2;
    return 0;
}

int32_t tq_jac(int32_t id, const double *x, double *vals) {
    (void)x;
    vals[0] = 1.0;
    vals[1] = 1.0;
    return 0;
}

int32_t tq_hess_structure(int32_t id, int32_t *rows, int32_t *cols) {
    int32_t N = getN(id);
    if (N < 0) return 1;
    for (int32_t i = 0; i < N; i++) {
        rows[i] = i + 1;
        cols[i] = i + 1;
    }
    return 0;
}

int32_t tq_hess(int32_t id, const double *x, const double *y, double obj_weight, double *vals) {
    (void)x; (void)y;
    int32_t N = getN(id);
    if (N < 0) return 1;
    for (int32_t i = 0; i < N; i++) vals[i] = 2.0 * obj_weight;
    return 0;
}

/* ── A second reference model: structured data only ───────────────────────
 * `sq` has NO one-integer constructor — instantiating it goes through the
 * schema + builder exclusively, which is the shape a producer emits whenever
 * the model needs more than a single size. Its schema declares two fields, so
 * a consumer's positional binding (arg1, arg2, ...) is exercised against
 * something wider than one field, and both scalar kinds and the array setter
 * are covered.
 *
 *   min  sum_i w_i (x_i - s)^2   s.t.  x_1 + x_2 = 1        (n >= 2)
 *
 * Schema order — which IS the argument order — is n, s, w.
 */
#define SQ_MAX_MODELS 16
#define SQ_MAX_N 64
static int32_t sq_n[SQ_MAX_MODELS];
static double sq_s[SQ_MAX_MODELS];
static double sq_w[SQ_MAX_MODELS][SQ_MAX_N];
static int32_t sq_models = 0;

static const char sq_schema_str[] =
    "{\"abi\":2,\"fields\":["
    "{\"name\":\"n\",\"kind\":\"scalar\",\"type\":\"i64\"},"
    "{\"name\":\"s\",\"kind\":\"scalar\",\"type\":\"f64\"},"
    "{\"name\":\"w\",\"kind\":\"array\",\"type\":\"f64\"}]}";

int32_t sq_schema(char *buf, int32_t cap) {
    int32_t len = (int32_t)(sizeof(sq_schema_str) - 1);
    for (int32_t i = 0; i < cap && i < len; i++) buf[i] = sq_schema_str[i];
    return len;
}

#define SQ_MAX_BUILDERS 16
static int64_t sqb_n[SQ_MAX_BUILDERS];
static double sqb_s[SQ_MAX_BUILDERS];
static double sqb_w[SQ_MAX_BUILDERS][SQ_MAX_N];
static int32_t sqb_wlen[SQ_MAX_BUILDERS];
static int32_t sqb_have[SQ_MAX_BUILDERS];      /* bit 0: n, bit 1: s, bit 2: w */
static int32_t sq_builders = 0;

int32_t sq_data_begin(void) {
    if (sq_builders >= SQ_MAX_BUILDERS) return 0;
    sqb_have[sq_builders] = 0;
    sqb_wlen[sq_builders] = 0;
    sq_builders++;
    return sq_builders;
}

int32_t sq_set_scalar_i64(int32_t b, const char *field, int64_t v) {
    if (b < 1 || b > sq_builders || !strsame(field, "n")) return 1;
    sqb_n[b - 1] = v;
    sqb_have[b - 1] |= 1;
    return 0;
}

int32_t sq_set_scalar_f64(int32_t b, const char *field, double v) {
    if (b < 1 || b > sq_builders || !strsame(field, "s")) return 1;
    sqb_s[b - 1] = v;
    sqb_have[b - 1] |= 2;
    return 0;
}

int32_t sq_set_array_f64(int32_t b, const char *field, const double *v, int32_t len) {
    if (b < 1 || b > sq_builders || !strsame(field, "w")) return 1;
    if (len < 0 || len > SQ_MAX_N) return 1;
    for (int32_t i = 0; i < len; i++) sqb_w[b - 1][i] = v[i];
    sqb_wlen[b - 1] = len;
    sqb_have[b - 1] |= 4;
    return 0;
}

/* Complete AND consistent: w must be as long as n says. */
int32_t sq_data_ready(int32_t b) {
    if (b < 1 || b > sq_builders) return 0;
    if (sqb_have[b - 1] != 7) return 0;
    return sqb_wlen[b - 1] == (int32_t)sqb_n[b - 1] ? 1 : 0;
}

int32_t sq_new_from_data(int32_t b) {
    if (sq_data_ready(b) != 1) return 0;
    int32_t n = (int32_t)sqb_n[b - 1];
    if (n < 2 || sq_models >= SQ_MAX_MODELS) return 0;
    sq_n[sq_models] = n;
    sq_s[sq_models] = sqb_s[b - 1];
    for (int32_t i = 0; i < n; i++) sq_w[sq_models][i] = sqb_w[b - 1][i];
    sq_models++;
    return sq_models;
}

static int32_t sq_getN(int32_t id) {
    return (id >= 1 && id <= sq_models) ? sq_n[id - 1] : -1;
}

int32_t sq_nvar(int32_t id) { return sq_getN(id); }
int32_t sq_ncon(int32_t id) { return sq_getN(id) > 0 ? 1 : -1; }
int32_t sq_nnzj(int32_t id) { return sq_getN(id) > 0 ? 2 : -1; }
int32_t sq_nnzh(int32_t id) { return sq_getN(id); }

int32_t sq_meta(int32_t id, double *x0, double *lvar, double *uvar, double *lcon, double *ucon) {
    int32_t N = sq_getN(id);
    if (N < 0) return 1;
    for (int32_t i = 0; i < N; i++) {
        x0[i] = 0.0;
        lvar[i] = -INFINITY;
        uvar[i] = INFINITY;
    }
    lcon[0] = 0.0;
    ucon[0] = 0.0;
    return 0;
}

int32_t sq_obj(int32_t id, const double *x, double *out) {
    int32_t N = sq_getN(id);
    if (N < 0) return 1;
    double s = 0.0;
    for (int32_t i = 0; i < N; i++) {
        double d = x[i] - sq_s[id - 1];
        s += sq_w[id - 1][i] * d * d;
    }
    *out = s;
    return 0;
}

int32_t sq_grad(int32_t id, const double *x, double *g) {
    int32_t N = sq_getN(id);
    if (N < 0) return 1;
    for (int32_t i = 0; i < N; i++)
        g[i] = 2.0 * sq_w[id - 1][i] * (x[i] - sq_s[id - 1]);
    return 0;
}

int32_t sq_cons(int32_t id, const double *x, double *c) {
    (void)id;
    c[0] = x[0] + x[1] - 1.0;
    return 0;
}

int32_t sq_jac_structure(int32_t id, int32_t *rows, int32_t *cols) {
    (void)id;
    rows[0] = 1; cols[0] = 1;
    rows[1] = 1; cols[1] = 2;
    return 0;
}

int32_t sq_jac(int32_t id, const double *x, double *vals) {
    (void)id; (void)x;
    vals[0] = 1.0;
    vals[1] = 1.0;
    return 0;
}

int32_t sq_hess_structure(int32_t id, int32_t *rows, int32_t *cols) {
    int32_t N = sq_getN(id);
    if (N < 0) return 1;
    for (int32_t i = 0; i < N; i++) {
        rows[i] = i + 1;
        cols[i] = i + 1;
    }
    return 0;
}

int32_t sq_hess(int32_t id, const double *x, const double *y, double obj_weight, double *vals) {
    (void)x; (void)y;
    int32_t N = sq_getN(id);
    if (N < 0) return 1;
    for (int32_t i = 0; i < N; i++) vals[i] = 2.0 * obj_weight * sq_w[id - 1][i];
    return 0;
}

/* ── `fx`: a FIXED model — no instantiation arguments ─────────────────────
 * The tq model at n = 3, exposed the way ExaModelsC compiles a core with no
 * placeholders: `fx_nargs()` reports 0, and `fx_new` keeps the one-integer C
 * signature but ignores its value. Everything after instantiation delegates
 * to the tq implementation (the ids share one model table). */
int32_t fx_nargs(void) { return 0; }
int32_t fx_new(int32_t n) { (void)n; return tq_new(3); }
int32_t fx_nvar(int32_t id) { return tq_nvar(id); }
int32_t fx_ncon(int32_t id) { return tq_ncon(id); }
int32_t fx_nnzj(int32_t id) { return tq_nnzj(id); }
int32_t fx_nnzh(int32_t id) { return tq_nnzh(id); }
int32_t fx_meta(int32_t id, double *x0, double *lvar, double *uvar, double *lcon, double *ucon) {
    return tq_meta(id, x0, lvar, uvar, lcon, ucon);
}
int32_t fx_obj(int32_t id, const double *x, double *out) { return tq_obj(id, x, out); }
int32_t fx_grad(int32_t id, const double *x, double *g) { return tq_grad(id, x, g); }
int32_t fx_cons(int32_t id, const double *x, double *c) { return tq_cons(id, x, c); }
int32_t fx_jac_structure(int32_t id, int32_t *rows, int32_t *cols) {
    return tq_jac_structure(id, rows, cols);
}
int32_t fx_jac(int32_t id, const double *x, double *vals) { return tq_jac(id, x, vals); }
int32_t fx_hess_structure(int32_t id, int32_t *rows, int32_t *cols) {
    return tq_hess_structure(id, rows, cols);
}
int32_t fx_hess(int32_t id, const double *x, const double *y, double obj_weight, double *vals) {
    return tq_hess(id, x, y, obj_weight, vals);
}
