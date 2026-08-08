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

static int32_t N = 0;

int32_t tq_init(int32_t n) {
    if (n < 2) return 1;
    N = n;
    return 0;
}

int32_t tq_nvar(void) { return N; }
int32_t tq_ncon(void) { return 1; }
int32_t tq_nnzj(void) { return 2; }
int32_t tq_nnzh(void) { return N; }

int32_t tq_meta(double *x0, double *lvar, double *uvar, double *lcon, double *ucon) {
    for (int32_t i = 0; i < N; i++) {
        x0[i] = 0.0;
        lvar[i] = -INFINITY;
        uvar[i] = INFINITY;
    }
    lcon[0] = 0.0;
    ucon[0] = 0.0;
    return 0;
}

int32_t tq_obj(const double *x, double *out) {
    double s = 0.0;
    for (int32_t i = 0; i < N; i++) {
        double d = x[i] - 1.0;
        s += d * d;
    }
    *out = s;
    return 0;
}

int32_t tq_grad(const double *x, double *g) {
    for (int32_t i = 0; i < N; i++) g[i] = 2.0 * (x[i] - 1.0);
    return 0;
}

int32_t tq_cons(const double *x, double *c) {
    c[0] = x[0] + x[1] - 1.0;
    return 0;
}

int32_t tq_jac_structure(int32_t *rows, int32_t *cols) {
    rows[0] = 1; cols[0] = 1;
    rows[1] = 1; cols[1] = 2;
    return 0;
}

int32_t tq_jac(const double *x, double *vals) {
    (void)x;
    vals[0] = 1.0;
    vals[1] = 1.0;
    return 0;
}

int32_t tq_hess_structure(int32_t *rows, int32_t *cols) {
    for (int32_t i = 0; i < N; i++) {
        rows[i] = i + 1;
        cols[i] = i + 1;
    }
    return 0;
}

int32_t tq_hess(const double *x, const double *y, double obj_weight, double *vals) {
    (void)x; (void)y;
    for (int32_t i = 0; i < N; i++) vals[i] = 2.0 * obj_weight;
    return 0;
}
