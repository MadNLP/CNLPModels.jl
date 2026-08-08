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
