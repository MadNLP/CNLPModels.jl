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
