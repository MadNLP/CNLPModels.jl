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

#define TQ_MAX_MODELS 64
static int32_t Ns[TQ_MAX_MODELS];
static int32_t n_models = 0;

/* Returns a positive model id, or 0 on failure. */
int32_t tq_new(int32_t n) {
    if (n < 2 || n_models >= TQ_MAX_MODELS) return 0;
    Ns[n_models++] = n;
    return n_models;
}

/* ── schema + builder ─────────────────────────────────────────────
 * The same model exposed through the structured-data surface, so consumers'
 * builder paths (named tuples, positional tuples) are testable against this
 * fixture too. One scalar field: n. */
static const char tq_schema_str[] =
    "{\"fields\":[{\"name\":\"n\",\"kind\":\"scalar\",\"type\":\"i64\"}]}";

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

/* ── What this library carries ─────────────────────────────────────────────
 * Library-level, not per-model: a consumer holding only a path has no prefix
 * to start from, so these two have fixed names. Optional, like the layout and
 * builder surfaces — a library that omits them is consumed exactly as before.
 */
/* The four complete models. The other prefixes in this file (bf, ns, nf, zz)
 * implement partial surfaces on purpose, to exercise failure paths — a
 * catalogue naming them would be advertising models that cannot be built. */
int32_t cnlp_nmodels(void) { return 4; }

int32_t cnlp_model_name(int32_t k, uint8_t *buf, int32_t cap) {
    static const char *names[4] = {"tq", "sq", "fx", "tb"};
    if (k < 0 || k >= 4) return -1;
    const char *nm = names[k];
    int32_t len = 0;
    while (nm[len]) len++;
    for (int32_t i = 0; i < cap && i < len; i++) buf[i] = (uint8_t)nm[i];
    return len;
}

/* ── Named blocks ─────────────────────────────────────────────────────────
 * The layout surface a compiled ExaModels library publishes: which named
 * variable, constraint and parameter blocks the model has, and where each one
 * sits.  `tq` names its variable `x`, its constraint `budget`, and carries a
 * two-element parameter block `w` — enough for a consumer to be exercised
 * against every kind without a Julia toolchain anywhere.
 *
 * Offsets are 0-based and lengths follow the instance, exactly as the
 * generated libraries report them.
 */
static double tq_w[TQ_MAX_MODELS][2];

int32_t tq_nblocks(int32_t id) { return getN(id) > 0 ? 3 : -1; }

int32_t tq_block_name(int32_t id, int32_t k, uint8_t *buf, int32_t cap) {
    const char *nm = k == 0 ? "x" : k == 1 ? "budget" : k == 2 ? "w" : 0;
    if (getN(id) <= 0 || !nm) return -1;
    int32_t len = 0;
    while (nm[len]) len++;
    for (int32_t i = 0; i < cap && i < len; i++) buf[i] = (uint8_t)nm[i];
    return len;
}

/* out: [kind, offset, length, ndims, dims...]; kind 0 = var, 1 = con, 2 = par */
int32_t tq_block(int32_t id, int32_t k, int32_t *out) {
    int32_t N = getN(id);
    if (N <= 0) return 1;
    if (k == 0)      { out[0] = 0; out[1] = 0; out[2] = N; out[3] = 1; out[4] = N; return 0; }
    else if (k == 1) { out[0] = 1; out[1] = 0; out[2] = 1; out[3] = 1; out[4] = 1; return 0; }
    else if (k == 2) { out[0] = 2; out[1] = 0; out[2] = 2; out[3] = 1; out[4] = 2; return 0; }
    return 1;
}

int32_t tq_get_value(int32_t id, int32_t k, double *vals, int32_t len) {
    if (getN(id) <= 0 || k != 2) return 1;
    if (len != 2) return 3;
    vals[0] = tq_w[id - 1][0];
    vals[1] = tq_w[id - 1][1];
    return 0;
}

int32_t tq_set_value(int32_t id, int32_t k, const double *vals, int32_t len) {
    if (getN(id) <= 0 || k != 2) return 1;
    if (len != 2) return 3;
    tq_w[id - 1][0] = vals[0];
    tq_w[id - 1][1] = vals[1];
    return 0;
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
    "{\"fields\":["
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

/* ── `tb`: a TABLE field through the builder ──────────────────────────────
 * min Σ_j w_j (x_{i_j} - 1)^2 over 3 variables, unconstrained — the one
 * schema field is a table `pts` with columns i (i64, 1-based variable index)
 * and w (f64, weight). Exercises the consumers' table path end-to-end. */
static const char tb_schema_str[] =
    "{\"fields\":[{\"name\":\"pts\",\"kind\":\"table\",\"columns\":"
    "[{\"name\":\"i\",\"type\":\"i64\"},{\"name\":\"w\",\"type\":\"f64\"}]}]}";
int32_t tb_schema(char *buf, int32_t cap) {
    int32_t len = (int32_t)(sizeof(tb_schema_str) - 1);
    for (int32_t k = 0; k < cap && k < len; k++) buf[k] = tb_schema_str[k];
    return len;
}
#define TB_MAX 8
#define TB_ROWS 8
static int64_t tb_bi[TB_MAX][TB_ROWS];
static double  tb_bw[TB_MAX][TB_ROWS];
static int32_t tb_ilen[TB_MAX], tb_wlen[TB_MAX];
static int32_t tb_nb = 0;
static int64_t tb_mi[TB_MAX][TB_ROWS];
static double  tb_mw[TB_MAX][TB_ROWS];
static int32_t tb_mrows[TB_MAX];
static int32_t tb_nm = 0;

int32_t tb_data_begin(void) { return tb_nb < TB_MAX ? ++tb_nb : 0; }
/* A column longer than the fixture can hold is refused with a nonzero
 * status — which is also how consumers' status-check paths get exercised. */
int32_t tb_set_col_i64(int32_t b, const char *field, const char *col, const int64_t *v, int32_t len) {
    (void)field; (void)col;
    if (b < 1 || b > tb_nb || len > TB_ROWS) return 1;
    for (int32_t k = 0; k < len; k++) tb_bi[b - 1][k] = v[k];
    tb_ilen[b - 1] = len;
    return 0;
}
int32_t tb_set_col_f64(int32_t b, const char *field, const char *col, const double *v, int32_t len) {
    (void)field; (void)col;
    if (b < 1 || b > tb_nb || len > TB_ROWS) return 1;
    for (int32_t k = 0; k < len; k++) tb_bw[b - 1][k] = v[k];
    tb_wlen[b - 1] = len;
    return 0;
}
int32_t tb_data_ready(int32_t b) {
    return (b >= 1 && b <= tb_nb && tb_ilen[b - 1] > 0 && tb_ilen[b - 1] == tb_wlen[b - 1]) ? 1 : 0;
}
int32_t tb_new_from_data(int32_t b) {
    if (!tb_data_ready(b) || tb_nm >= TB_MAX) return 0;
    int32_t m = tb_nm++;
    tb_mrows[m] = tb_ilen[b - 1];
    for (int32_t k = 0; k < tb_mrows[m]; k++) {
        tb_mi[m][k] = tb_bi[b - 1][k];
        tb_mw[m][k] = tb_bw[b - 1][k];
    }
    return m + 1;
}
int32_t tb_nvar(int32_t id) { return (id >= 1 && id <= tb_nm) ? 3 : -1; }
int32_t tb_ncon(int32_t id) { return (id >= 1 && id <= tb_nm) ? 0 : -1; }
int32_t tb_nnzj(int32_t id) { return (id >= 1 && id <= tb_nm) ? 0 : -1; }
int32_t tb_nnzh(int32_t id) { return (id >= 1 && id <= tb_nm) ? 3 : -1; }
int32_t tb_meta(int32_t id, double *x0, double *lvar, double *uvar, double *lcon, double *ucon) {
    (void)lcon; (void)ucon;
    if (id < 1 || id > tb_nm) return 1;
    for (int32_t k = 0; k < 3; k++) { x0[k] = 0.0; lvar[k] = -HUGE_VAL; uvar[k] = HUGE_VAL; }
    return 0;
}
int32_t tb_obj(int32_t id, const double *x, double *out) {
    if (id < 1 || id > tb_nm) return 1;
    double s = 0.0;
    for (int32_t k = 0; k < tb_mrows[id - 1]; k++) {
        double d = x[tb_mi[id - 1][k] - 1] - 1.0;
        s += tb_mw[id - 1][k] * d * d;
    }
    *out = s;
    return 0;
}
int32_t tb_grad(int32_t id, const double *x, double *g) {
    if (id < 1 || id > tb_nm) return 1;
    for (int32_t k = 0; k < 3; k++) g[k] = 0.0;
    for (int32_t k = 0; k < tb_mrows[id - 1]; k++)
        g[tb_mi[id - 1][k] - 1] += 2.0 * tb_mw[id - 1][k] * (x[tb_mi[id - 1][k] - 1] - 1.0);
    return 0;
}
int32_t tb_cons(int32_t id, const double *x, double *c) {
    (void)x; (void)c;
    return (id >= 1 && id <= tb_nm) ? 0 : 1;
}
int32_t tb_jac_structure(int32_t id, int32_t *rows, int32_t *cols) {
    (void)rows; (void)cols;
    return (id >= 1 && id <= tb_nm) ? 0 : 1;
}
int32_t tb_jac(int32_t id, const double *x, double *vals) {
    (void)x; (void)vals;
    return (id >= 1 && id <= tb_nm) ? 0 : 1;
}
int32_t tb_hess_structure(int32_t id, int32_t *rows, int32_t *cols) {
    if (id < 1 || id > tb_nm) return 1;
    for (int32_t k = 0; k < 3; k++) { rows[k] = k + 1; cols[k] = k + 1; }
    return 0;
}
int32_t tb_hess(int32_t id, const double *x, const double *y, double obj_weight, double *vals) {
    (void)x; (void)y;
    if (id < 1 || id > tb_nm) return 1;
    for (int32_t k = 0; k < 3; k++) vals[k] = 0.0;
    for (int32_t k = 0; k < tb_mrows[id - 1]; k++)
        vals[tb_mi[id - 1][k] - 1] += 2.0 * obj_weight * tb_mw[id - 1][k];
    return 0;
}

/* ── degenerate surfaces, one consumer failure path each ────────────────── */
/* `ns`: a builder entry point without a schema — nothing to bind against. */
int32_t ns_data_begin(void) { return 1; }

/* `bf`: schema + builder whose data_begin fails. */
static const char bf_schema_str[] =
    "{\"fields\":[{\"name\":\"n\",\"kind\":\"scalar\",\"type\":\"i64\"}]}";
int32_t bf_schema(char *buf, int32_t cap) {
    int32_t len = (int32_t)(sizeof(bf_schema_str) - 1);
    for (int32_t k = 0; k < cap && k < len; k++) buf[k] = bf_schema_str[k];
    return len;
}
int32_t bf_data_begin(void) { return 0; }

/* `nf`: the builder runs to the end and new_from_data fails. */
static const char nf_schema_str[] =
    "{\"fields\":[{\"name\":\"n\",\"kind\":\"scalar\",\"type\":\"i64\"}]}";
int32_t nf_schema(char *buf, int32_t cap) {
    int32_t len = (int32_t)(sizeof(nf_schema_str) - 1);
    for (int32_t k = 0; k < cap && k < len; k++) buf[k] = nf_schema_str[k];
    return len;
}
int32_t nf_data_begin(void) { return 1; }
int32_t nf_set_scalar_i64(int32_t b, const char *f, int64_t v) { (void)b; (void)f; (void)v; return 0; }
int32_t nf_data_ready(int32_t b) { (void)b; return 1; }
int32_t nf_new_from_data(int32_t b) { (void)b; return 0; }

/* `zz`: a fixed library whose constructor fails. */
int32_t zz_nargs(void) { return 0; }
int32_t zz_new(int32_t n) { (void)n; return 0; }

/* Integer weights are accepted and widened: the schema's KIND (array) is the
 * contract; an integer literal on the caller's side is not a different field. */
int32_t sq_set_array_i64(int32_t b, const char *field, const int64_t *v, int32_t len) {
    if (b < 1 || b > sq_builders || !strsame(field, "w")) return 1;
    if (len < 0 || len > SQ_MAX_N) return 1;
    for (int32_t i = 0; i < len; i++) sqb_w[b - 1][i] = (double)v[i];
    sqb_wlen[b - 1] = len;
    sqb_have[b - 1] |= 4;
    return 0;
}
