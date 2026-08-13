"""
    CNLPModels

Consume an NLP exposed by a shared library through a plain C interface as an
`NLPModels.AbstractNLPModel`, so any NLPModels-compatible solver (MadNLP,
Ipopt, ...) can solve it. The main producer of such libraries is ExaModels'
tape recorder (`compile_library`), but any
library implementing the ABI below works, in any language.

# C ABI convention

For a chosen symbol `prefix` (e.g. `rec`), the library exposes — all indices
1-based, Hessian as the lower triangle of `obj_weight * ∇²f + Σ yᵢ ∇²cᵢ`,
every function returning `Cint` status `0` on success:

    <prefix>_new(n::Cint)::Cint                   # instantiate at size n; returns
                                                  # a positive model id, 0 on failure
    <prefix>_nvar(id::Cint)::Cint
    <prefix>_ncon(id::Cint)::Cint
    <prefix>_nnzj(id::Cint)::Cint
    <prefix>_nnzh(id::Cint)::Cint
    <prefix>_meta(id, x0, lvar, uvar, lcon, ucon)::Cint      # Ptr{Cdouble} × 5
    <prefix>_obj(id, x::Ptr{Cdouble}, out::Ptr{Cdouble})::Cint
    <prefix>_grad(id, x::Ptr{Cdouble}, g::Ptr{Cdouble})::Cint
    <prefix>_cons(id, x::Ptr{Cdouble}, c::Ptr{Cdouble})::Cint
    <prefix>_jac_structure(id, rows::Ptr{Cint}, cols::Ptr{Cint})::Cint
    <prefix>_jac(id, x::Ptr{Cdouble}, vals::Ptr{Cdouble})::Cint
    <prefix>_hess_structure(id, rows::Ptr{Cint}, cols::Ptr{Cint})::Cint
    <prefix>_hess(id, x::Ptr{Cdouble}, y::Ptr{Cdouble}, obj_weight::Cdouble,
                  vals::Ptr{Cdouble})::Cint

Handles make models independent: any number of `CNLPModel`s may coexist per
library (each `<prefix>_new` call creates one; models live for the process
lifetime). A library may also carry *several models*, one prefix each — name
the one you want as a symbol, `CNLPModel("@grid", :acopf, ...)`.

# The libblastrampoline seam

A `juliac`-compiled library carries its own (privatized) Julia runtime, but
libblastrampoline is resolved once per process by soname, so it is *shared*
with the host. The library's runtime initializes lazily on its first call and
re-forwards the shared trampoline to its own BLAS, silently clearing the
host's forwarding (symptom: `Error: no BLAS/LAPACK library loaded for
dgemm_()` and solvers degrading to garbage). [`load`](@ref) snapshots the
host's BLAS configuration before `dlopen`, and [`CNLPModel`](@ref) restores it
after the library's first call; [`restore_blas!`](@ref) can be called again at
any time.

# Usage

```julia
using CNLPModels, MadNLP
lib = CNLPModels.load("librecorder.so")
m = CNLPModel(lib, 1000; prefix = "rec")   # rec_new(1000) → id, fixes BLAS
res = madnlp(m)
```

Instantiation arguments are positional, one per schema field — the same
spelling `ExaModel(core, arg1, arg2, ...)` uses on the producer side.
"""
module CNLPModels

using Libdl
using LinearAlgebra
import NLPModels: NLPModels, AbstractNLPModel, NLPModelMeta, Counters,
    increment!, @lencheck

export CNLPModel, restore_blas!, schema_json, set_path!, @cnlp_str

include("patchversion.jl")
include("private_runtime.jl")

"""
    CLib

Handle to a loaded NLP shared library: the `dlopen` handle plus the snapshot
of the host's BLAS forwarding taken *before* the library was loaded. Create
with [`load`](@ref).
"""
struct CLib
    path::String
    handle::Ptr{Cvoid}
    blas_libs::Vector{String}
end

# ── library path registry ────────────────────────────────────────────────────

const _PATHS = String[]
const _LIBS = Dict{String, CLib}()

"""
    set_path!(dirs...)

Set the library search path used by name-based loading (`CNLPModels.lib("acopf")`,
`CNLPModel("@acopf", ...)`, `cnlp"acopf"`). Initialized from the colon-separated
`CNLPMODELS_PATH` environment variable; calling `set_path!` replaces it.
"""
function set_path!(dirs::AbstractString...)
    empty!(_PATHS)
    append!(_PATHS, String.(dirs))
    return copy(_PATHS)
end

function _paths()
    if isempty(_PATHS)
        env = get(ENV, "CNLPMODELS_PATH", "")
        isempty(env) || append!(_PATHS, split(env, ":"))
    end
    return _PATHS
end

"""
    lib(name::AbstractString) -> CLib

Resolve `lib<name>.<dlext>` against the search path ([`set_path!`](@ref) /
`CNLPMODELS_PATH`) — also accepting `<dir>/lib/lib<name>.<dlext>`, the layout
`compile_library` produces — load it, and cache the handle by name.
"""
function lib(name::AbstractString)
    get!(_LIBS, String(name)) do
        fname = "lib" * name * "." * Libdl.dlext
        for d in _paths(), cand in (joinpath(d, fname), joinpath(d, name, "lib", fname), joinpath(d, "lib", fname))
            isfile(cand) && return load(cand)
        end
        error("library $fname not found on the CNLPModels path " *
              "($(isempty(_paths()) ? "empty — call set_path! or set CNLPMODELS_PATH" : join(_paths(), ':')))")
    end
end

"""
    cnlp"name"

The shortcut onto the library path: resolve `lib<name>.<dlext>` against
[`set_path!`](@ref) / `CNLPMODELS_PATH`, check that it exists and loads as a
shared library, and return the (cached) handle — usable directly as the
library argument of [`CNLPModel`](@ref):

    m = CNLPModel(cnlp"acopf", bus, vmin, 100.0)

Resolution and validation happen at run time (never at parse time), so the
literal is safe inside precompiled code.
"""
macro cnlp_str(name)
    return :(lib($name))
end

"""
    load(path::AbstractString) -> CLib

Snapshot the host's BLAS configuration, then `dlopen` the library with
`RTLD_LOCAL | RTLD_DEEPBIND` (so the library prefers its own bundled
dependencies and leaks no symbols into the host).

An **unbundled** juliac library — one linked against the standard
`libjulia` rather than carrying a privatized copy — cannot be loaded
into a Julia process as-is: sharing the host's runtime, its first call
aborts the whole process. On Linux, `load` detects this and gives the
library a private copy of the *installed* runtime instead (patched in
scratch, one per library file, cached), after which it behaves exactly
like a bundled one. On other platforms the unbundled form is refused
with an explanation — compile with `bundle = true` there. The Julia
running here must match the version the library was linked against.
"""
function load(path::AbstractString)
    blas_libs = [l.libname for l in BLAS.get_config().loaded_libs]
    handle = _dlopen_model(String(path), Libdl.RTLD_LOCAL | Libdl.RTLD_DEEPBIND)
    return CLib(String(path), handle, blas_libs)
end

"""
    restore_blas!(lib::CLib)

Re-forward the host's libblastrampoline to the BLAS/LAPACK libraries that were
active before `lib` was loaded. Idempotent; called automatically by
[`CNLPModel`](@ref) after the library's first call.
"""
function restore_blas!(lib::CLib)
    for (i, l) in enumerate(lib.blas_libs)
        BLAS.lbt_forward(l; clear = (i == 1))
    end
    return lib
end

struct CNLPModel <: AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::Counters
    lib::CLib
    id::Cint
    obj_p::Ptr{Cvoid}
    grad_p::Ptr{Cvoid}
    cons_p::Ptr{Cvoid}
    jac_structure_p::Ptr{Cvoid}
    jac_p::Ptr{Cvoid}
    hess_structure_p::Ptr{Cvoid}
    hess_p::Ptr{Cvoid}
    # cached Jacobian structure + a value buffer, for jprod!/jtprod!
    jrows::Vector{Int}
    jcols::Vector{Int}
    jbuf::Vector{Float64}
end

@noinline _status_error(sym, st) =
    error("$sym returned nonzero status $st")

@inline function _check(st::Cint, sym::Symbol)
    st == 0 || _status_error(sym, st)
    return nothing
end

"""
    schema_json(lib::CLib; prefix = "rec") -> String
    schema_json(lib::CLib, model::Symbol) -> String

The library's data schema as published by `<prefix>_schema` (ABI v2,
structured libraries only): a JSON description of the fields, kinds
(scalar/array/table) and column types. In a library carrying several models
the schema is per model — name it as a symbol, exactly as in [`CNLPModel`](@ref).
"""
function schema_json(lib::CLib; prefix::AbstractString = "rec")
    fp = Libdl.dlsym(lib.handle, Symbol(prefix, "_schema"))
    n = ccall(fp, Cint, (Ptr{UInt8}, Cint), C_NULL, Cint(0))
    buf = Vector{UInt8}(undef, Int(n))
    ccall(fp, Cint, (Ptr{UInt8}, Cint), buf, n)
    return String(buf)
end

schema_json(lib::CLib, model::Symbol) = schema_json(lib; prefix = String(model))

# Top-level field names from the schema JSON, in declaration order. The
# schema is this package's own contract (see the README), so a structural
# scan suffices: field objects sit at bracket depth 2 inside "fields"
# (table columns at depth 4 are thereby excluded).
function _schema_field_names(json::AbstractString)
    names = String[]
    r = findfirst("\"fields\"", json)
    r === nothing && error("library schema has no \"fields\" array")
    depth = 0
    i = last(r)
    while i < lastindex(json)
        i = nextind(json, i)
        ch = json[i]
        if ch == '[' || ch == '{'
            depth += 1
        elseif ch == ']' || ch == '}'
            depth -= 1
            depth <= 0 && break
        elseif ch == '"'
            j = findnext('"', json, nextind(json, i))
            tok = json[nextind(json, i):prevind(json, j)]
            if depth == 2 && tok == "name"
                k1 = findnext('"', json, nextind(json, j))
                k2 = findnext('"', json, nextind(json, k1))
                push!(names, json[nextind(json, k1):prevind(json, k2)])
                i = k2
            else
                i = j
            end
        end
    end
    return names
end

# ── Instantiation: positional arguments → model id ───────────────────────────
#
# Arguments are positional — one value per schema field, in the order the
# library publishes them. That is deliberately the same convention the producer
# side uses: an `ExaCore` built against placeholders is instantiated as
# `ExaModel(core, arg1, arg2, ...)` and compiled as
# `compile_library(out, core, arg1, ...)`, so a compiled model is consumed the
# way it was written. The schema a compiled recipe publishes names its fields
# `arg1`, `arg2`, ... for exactly that reason.
#
# A field's value is a scalar, an array, or a table (a vector of named tuples,
# sent columnar) — the same grammar the producer derives its schema from.
#
# A lone integer takes the one-knob `<prefix>_new(n)` when the library exports
# it, and falls through to the builder otherwise. The two surfaces are disjoint
# in practice: `compile_library` emits `P_new` and no builder precisely when the
# schema is a single integer scalar.

function _instantiate(lib::CLib, prefix::AbstractString, args::Tuple)
    if length(args) == 1 && args[1] isa Integer
        p = Libdl.dlsym(lib.handle, Symbol(prefix, "_new"); throw_error = false)
        if p !== nothing
            id = ccall(p, Cint, (Cint,), Cint(args[1]))
            id > 0 || _status_error(Symbol(prefix, "_new"), id)
            return id
        end
    end
    # A FIXED model — a library whose `<prefix>_nargs()` reports 0 — consumes
    # no instantiation data: `<prefix>_new` keeps its one-integer C signature
    # but ignores the value, so no arguments at all is the natural call. A
    # library that does not declare its arity keeps the old behaviour and
    # falls through to the builder surface (where a schema with no fields is
    # the other legitimate "no instance data" case).
    if isempty(args)
        q = Libdl.dlsym(lib.handle, Symbol(prefix, "_nargs"); throw_error = false)
        if q !== nothing && ccall(q, Cint, ()) == 0
            p = Libdl.dlsym(lib.handle, Symbol(prefix, "_new"))
            id = ccall(p, Cint, (Cint,), Cint(0))
            id > 0 || _status_error(Symbol(prefix, "_new"), id)
            return id
        end
    end
    return _fill_data(lib, prefix, _bind(lib, prefix, args))
end

# Positional arguments against the schema's field order. No arguments at all is
# the "no instance data" case, and is checked the same way — a schema declaring
# fields says so here, rather than the library reporting itself incomplete
# several calls later.
function _bind(lib::CLib, prefix::AbstractString, args::Tuple)
    Libdl.dlsym(lib.handle, Symbol(prefix, "_data_begin"); throw_error = false) === nothing &&
        error("this library has no builder surface: it instantiates from a " *
              "single integer, `CNLPModel(lib, n)`")
    names = _schema_field_names(schema_json(lib; prefix = prefix))
    length(names) == length(args) || error(
        "given $(length(args)) argument$(length(args) == 1 ? "" : "s") but the " *
        "library's schema declares $(length(names)) field" *
        "$(length(names) == 1 ? "" : "s")" *
        (isempty(names) ? "" : ": " * join(names, ", ")))
    return NamedTuple{Tuple(Symbol.(names))}(args)
end

# One builder field, by value dispatch: numbers are scalars, numeric vectors
# are arrays, vectors of named tuples are tables (sent column-by-column). The
# library itself validates names, kinds, completeness and column lengths.
_push_field(fp, sym, b, f, val::AbstractFloat) =
    _check(ccall(fp(:set_scalar_f64), Cint, (Cint, Cstring, Cdouble),
        b, f, Cdouble(val)), sym(:set_scalar_f64))
_push_field(fp, sym, b, f, val::Integer) =
    _check(ccall(fp(:set_scalar_i64), Cint, (Cint, Cstring, Clonglong),
        b, f, Clonglong(val)), sym(:set_scalar_i64))
_push_field(fp, sym, b, f, val::AbstractVector{<:AbstractFloat}) =
    _check(ccall(fp(:set_array_f64), Cint, (Cint, Cstring, Ptr{Cdouble}, Cint),
        b, f, convert(Vector{Float64}, val), Cint(length(val))), sym(:set_array_f64))
_push_field(fp, sym, b, f, val::AbstractVector{<:Integer}) =
    _check(ccall(fp(:set_array_i64), Cint, (Cint, Cstring, Ptr{Clonglong}, Cint),
        b, f, convert(Vector{Int64}, val), Cint(length(val))), sym(:set_array_i64))
function _push_field(fp, sym, b, f, val::AbstractVector{<:NamedTuple})
    for cn in fieldnames(eltype(val))
        _push_col(fp, sym, b, f, string(cn), [getfield(r, cn) for r in val])
    end
end
_push_field(fp, sym, b, f, val) =
    error("unsupported field $f::$(typeof(val)): fields must be numbers, " *
          "numeric vectors, or vectors of named tuples of numbers")

_push_col(fp, sym, b, f, c, col::AbstractVector{<:AbstractFloat}) =
    _check(ccall(fp(:set_col_f64), Cint, (Cint, Cstring, Cstring, Ptr{Cdouble}, Cint),
        b, f, c, convert(Vector{Float64}, col), Cint(length(col))), sym(:set_col_f64))
_push_col(fp, sym, b, f, c, col::AbstractVector{<:Integer}) =
    _check(ccall(fp(:set_col_i64), Cint, (Cint, Cstring, Cstring, Ptr{Clonglong}, Cint),
        b, f, c, convert(Vector{Int64}, col), Cint(length(col))), sym(:set_col_i64))
_push_col(fp, sym, b, f, c, col) =
    error("unsupported column type $(eltype(col)) for $f.$c")

function _fill_data(lib::CLib, prefix::AbstractString, data::NamedTuple)
    sym(s) = Symbol(prefix, "_", s)
    fp(s) = Libdl.dlsym(lib.handle, sym(s))
    b = ccall(fp(:data_begin), Cint, ())
    b > 0 || _status_error(sym(:data_begin), b)
    for (fname, val) in pairs(data)
        _push_field(fp, sym, b, string(fname), val)
    end
    ccall(fp(:data_ready), Cint, (Cint,), b) == 1 ||
        error("library reports data incomplete or inconsistent after all fields were set")
    id = ccall(fp(:new_from_data), Cint, (Cint,), b)
    id > 0 || _status_error(sym(:new_from_data), id)
    return id
end

"""
    CNLPModel(lib::CLib, arg1, arg2, ...; prefix = "rec", name = basename(lib.path))

Create a model instance from `lib` and wrap it as an `AbstractNLPModel`.

The arguments are the values the model is instantiated with — one per field of
the library's schema, positionally, in the order the library publishes them.
This is the same spelling the producer side uses (`ExaModel(core, arg1, ...)`,
`compile_library(out, core, arg1, ...)`), so a compiled model is consumed the
way it was written.

Each value is a **number**, a **numeric vector**, or a **table** (a vector of
named tuples, sent to the ABI v2 builder column by column and validated against
the library's schema). A lone integer uses `<prefix>_new(n)` when the library
exports it and the builder otherwise; with no arguments at all the model is
built from no instance data, which is valid when the schema declares no fields.

Any number of instances may coexist per library. The first library call lazily
finishes its runtime initialization, after which the host's BLAS forwarding is
restored — see [`restore_blas!`](@ref).

    m = CNLPModel(lib, 1000; prefix = "rosen")             # rosen_new(1000)
    m = CNLPModel(lib, [(i = 1, pd = 0.4)], [0.9], 100.0)  # table, array, scalar
"""
function CNLPModel(
    lib::CLib,
    args...;
    prefix::AbstractString = "rec",
    name::AbstractString = basename(lib.path),
)
    sym(s) = Symbol(prefix, "_", s)
    fp(s) = Libdl.dlsym(lib.handle, sym(s))

    id = _instantiate(lib, prefix, args)
    # The library's runtime is fully initialized by now: repair whatever its
    # lazy initialization did to the shared trampoline.
    restore_blas!(lib)

    nvar = Int(ccall(fp(:nvar), Cint, (Cint,), id))
    ncon = Int(ccall(fp(:ncon), Cint, (Cint,), id))
    nnzj = Int(ccall(fp(:nnzj), Cint, (Cint,), id))
    nnzh = Int(ccall(fp(:nnzh), Cint, (Cint,), id))

    x0 = zeros(nvar); lvar = zeros(nvar); uvar = zeros(nvar)
    lcon = zeros(ncon); ucon = zeros(ncon)
    st = ccall(fp(:meta), Cint,
        (Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
        id, x0, lvar, uvar, lcon, ucon)
    _check(st, sym(:meta))

    meta = NLPModelMeta(
        nvar; ncon = ncon, nnzj = nnzj, nnzh = nnzh,
        x0 = x0, lvar = lvar, uvar = uvar, lcon = lcon, ucon = ucon,
        minimize = true, name = String(name),
    )
    m = CNLPModel(
        meta, Counters(), lib, id,
        fp(:obj), fp(:grad), fp(:cons),
        fp(:jac_structure), fp(:jac), fp(:hess_structure), fp(:hess),
        Vector{Int}(undef, nnzj), Vector{Int}(undef, nnzj),
        Vector{Float64}(undef, nnzj),
    )
    NLPModels.jac_structure!(m, m.jrows, m.jcols)
    return m
end

"""
    CNLPModel(spec::AbstractString, arg1, arg2, ...; kwargs...)

String-based construction:

  - `"@opf"` resolves the **name** `opf` via [`lib`](@ref) against
    [`set_path!`](@ref) / `CNLPMODELS_PATH` (`\$dir/libopf.so`, or the
    bundled layouts `\$dir/opf/lib/libopf.so` and `\$dir/lib/libopf.so`);
    the prefix defaults to the name;
  - any other string is a filesystem **path**, relative to the current
    directory or absolute, exactly as written — either the shared library
    itself, or a bundle directory `compile_library` produced, in which case
    the library is found inside it. The prefix defaults to the resolved
    name stripped of `lib` and the extension (`.../librosen.so` → `"rosen"`,
    `/path/to/opf/` → `"opf"`).

Override either default with `prefix=`.

    m = CNLPModel("@acopf", bus, vmin, 100.0)     # search path
    m = CNLPModel("rosen", 1000)                  # ./rosen (file or bundle dir)
    m = CNLPModel("/opt/models/rosen", 1000)      # full path
"""
function CNLPModel(
    spec::AbstractString, args...;
    prefix::AbstractString = _default_prefix(spec), kwargs...,
)
    return CNLPModel(_resolve_spec(spec), args...; prefix = prefix, kwargs...)
end

"""
    CNLPModel(lib, model::Symbol, arg1, arg2, ...; name = String(model))

Select a model **by name** in a library carrying several.

One shared library may export any number of models, each under its own symbol
prefix, each with its own schema and its own instances — `:acopf` binds
`acopf_new`, `acopf_obj`, `acopf_meta`, ... So the symbol *is* the prefix, and
naming it is the whole of model selection:

    m = CNLPModel("@grid", :acopf, bus, 100.0)   # acopf_* inside libgrid.so
    d = CNLPModel("@grid", :dcopf, bus)          # dcopf_* in the same library

`lib` is anything the single-model constructors accept — a `CLib`, a `cnlp"..."`
literal, an `"@name"`, or a path. Models in one library share the library's
address space and, for a library carrying a language runtime, its single runtime
copy; they are otherwise independent, and instances of different models coexist
exactly as instances of one model do.

Omitting the symbol keeps the single-model spelling, where the prefix falls back
to the library name — so a one-model library is unaffected.
"""
function CNLPModel(
    lib::CLib, model::Symbol, args...; name::AbstractString = String(model),
)
    _require_model(lib, model)
    return CNLPModel(lib, args...; prefix = String(model), name = name)
end

# A name this library does not carry must be reported HERE. `_nvar` is the
# witness because the ABI requires it of every model whichever way the model is
# instantiated — unlike `_new` (absent from builder-only models) or
# `_data_begin` (absent from one-knob ones). Without this check a mistyped name
# surfaces several calls later as "this library has no builder surface", which
# describes a single-model library rather than a name that is not in this one.
function _require_model(lib::CLib, model::Symbol)
    Libdl.dlsym(lib.handle, Symbol(model, "_nvar"); throw_error = false) === nothing &&
        error("library $(lib.path) carries no model named `$model` " *
              "(it exports no `$(model)_nvar`)")
    return nothing
end

function CNLPModel(spec::AbstractString, model::Symbol, args...; kwargs...)
    return CNLPModel(_resolve_spec(spec), model, args...; kwargs...)
end

# `@name` resolves on the search path; any other string is a filesystem path,
# relative to the current directory or absolute, exactly as written.
_is_name(spec::AbstractString) = startswith(spec, "@")

_default_prefix(spec::AbstractString) =
    _is_name(spec) ? String(spec[2:end]) : _prefix_from_path(spec)

# `librosen.so` → `rosen`; a bundle directory or a file not following the
# `lib<name>` convention keeps its stem, and `prefix=` remains the override
# for libraries whose symbols are named independently of the file.
function _prefix_from_path(path::AbstractString)
    base = first(splitext(basename(rstrip(path, '/'))))
    return startswith(base, "lib") && length(base) > 3 ? base[4:end] : base
end

# A path names a shared library directly, or a bundle DIRECTORY — the layout
# `compile_library` produces — in which case the library is found inside it.
# Returned ABSOLUTE: `dlopen` treats a slash-free relative like `qp.so` as a
# soname to search the system path for, not as a file in the current
# directory — it resolved locally only by environmental accident and failed
# in CI.
function _resolve_path(spec::AbstractString)
    isfile(spec) && return abspath(spec)
    if isdir(spec)
        fname = "lib" * basename(rstrip(spec, '/')) * "." * Libdl.dlext
        for cand in (joinpath(spec, "lib", fname), joinpath(spec, fname))
            isfile(cand) && return abspath(cand)
        end
        error("no shared library in $spec (tried lib/$fname and $fname)")
    end
    error("no shared library at $spec")
end

# Cached like name-resolution: absolute-path keys cannot collide with bare
# names, so the one registry serves both.
_resolve_spec(spec::AbstractString) =
    _is_name(spec) ? lib(spec[2:end]) :
    get!(() -> load(_resolve_path(spec)), _LIBS, abspath(spec))

function NLPModels.obj(m::CNLPModel, x::AbstractVector{Float64})
    @lencheck m.meta.nvar x
    increment!(m, :neval_obj)
    out = Ref{Cdouble}(0.0)
    _check(ccall(m.obj_p, Cint, (Cint, Ptr{Cdouble}, Ptr{Cdouble}), m.id, x, out), :obj)
    return out[]
end

function NLPModels.grad!(m::CNLPModel, x::AbstractVector{Float64}, g::AbstractVector{Float64})
    @lencheck m.meta.nvar x g
    increment!(m, :neval_grad)
    _check(ccall(m.grad_p, Cint, (Cint, Ptr{Cdouble}, Ptr{Cdouble}), m.id, x, g), :grad)
    return g
end

function NLPModels.cons!(m::CNLPModel, x::AbstractVector{Float64}, c::AbstractVector{Float64})
    @lencheck m.meta.nvar x
    @lencheck m.meta.ncon c
    increment!(m, :neval_cons)
    _check(ccall(m.cons_p, Cint, (Cint, Ptr{Cdouble}, Ptr{Cdouble}), m.id, x, c), :cons)
    return c
end

function NLPModels.jac_structure!(
    m::CNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer},
)
    @lencheck m.meta.nnzj rows cols
    r = Vector{Cint}(undef, m.meta.nnzj)
    c = Vector{Cint}(undef, m.meta.nnzj)
    _check(ccall(m.jac_structure_p, Cint, (Cint, Ptr{Cint}, Ptr{Cint}), m.id, r, c), :jac_structure)
    copyto!(rows, r); copyto!(cols, c)
    return rows, cols
end

function NLPModels.jac_coord!(m::CNLPModel, x::AbstractVector{Float64}, vals::AbstractVector{Float64})
    @lencheck m.meta.nvar x
    @lencheck m.meta.nnzj vals
    increment!(m, :neval_jac)
    _check(ccall(m.jac_p, Cint, (Cint, Ptr{Cdouble}, Ptr{Cdouble}), m.id, x, vals), :jac)
    return vals
end

function NLPModels.hess_structure!(
    m::CNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer},
)
    @lencheck m.meta.nnzh rows cols
    r = Vector{Cint}(undef, m.meta.nnzh)
    c = Vector{Cint}(undef, m.meta.nnzh)
    _check(ccall(m.hess_structure_p, Cint, (Cint, Ptr{Cint}, Ptr{Cint}), m.id, r, c), :hess_structure)
    copyto!(rows, r); copyto!(cols, c)
    return rows, cols
end

for (fn, inc) in ((:jprod_nln!, :neval_jprod), (:jprod!, :neval_jprod))
    @eval function NLPModels.$fn(
        m::CNLPModel, x::AbstractVector{Float64}, v::AbstractVector{Float64},
        Jv::AbstractVector{Float64},
    )
        @lencheck m.meta.nvar x v
        @lencheck m.meta.ncon Jv
        increment!(m, $(QuoteNode(inc)))
        NLPModels.jac_coord!(m, x, m.jbuf)
        fill!(Jv, 0.0)
        @inbounds for k in eachindex(m.jbuf)
            Jv[m.jrows[k]] += m.jbuf[k] * v[m.jcols[k]]
        end
        return Jv
    end
end

for (fn, inc) in ((:jtprod_nln!, :neval_jtprod), (:jtprod!, :neval_jtprod))
    @eval function NLPModels.$fn(
        m::CNLPModel, x::AbstractVector{Float64}, v::AbstractVector{Float64},
        Jtv::AbstractVector{Float64},
    )
        @lencheck m.meta.nvar x Jtv
        @lencheck m.meta.ncon v
        increment!(m, $(QuoteNode(inc)))
        NLPModels.jac_coord!(m, x, m.jbuf)
        fill!(Jtv, 0.0)
        @inbounds for k in eachindex(m.jbuf)
            Jtv[m.jcols[k]] += m.jbuf[k] * v[m.jrows[k]]
        end
        return Jtv
    end
end

function NLPModels.hess_coord!(
    m::CNLPModel, x::AbstractVector{Float64}, y::AbstractVector{Float64},
    vals::AbstractVector{Float64}; obj_weight::Real = 1.0,
)
    @lencheck m.meta.nvar x
    @lencheck m.meta.ncon y
    @lencheck m.meta.nnzh vals
    increment!(m, :neval_hess)
    _check(ccall(m.hess_p, Cint,
        (Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Ptr{Cdouble}),
        m.id, x, y, Cdouble(obj_weight), vals), :hess)
    return vals
end

function NLPModels.hess_coord!(
    m::CNLPModel, x::AbstractVector{Float64}, vals::AbstractVector{Float64};
    obj_weight::Real = 1.0,
)
    return NLPModels.hess_coord!(m, x, zeros(m.meta.ncon), vals; obj_weight = obj_weight)
end

end # module CNLPModels
