"""
    CNLPModels

Consume an NLP exposed by a shared library through a plain C interface as an
`NLPModels.AbstractNLPModel`, so any NLPModels-compatible solver (MadNLP,
Ipopt, ...) can solve it. The main producer of such libraries is ExaModels'
recorder (`record`/`replay`) AOT-compiled with `juliac --output-lib`, but any
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
lifetime).

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
m = CNLPModel(lib; prefix = "rec", n = 1000)   # rec_new(1000) → id, fixes BLAS
res = madnlp(m)
```
"""
module CNLPModels

using Libdl
using LinearAlgebra
import NLPModels: NLPModels, AbstractNLPModel, NLPModelMeta, Counters,
    increment!, @lencheck

export CNLPModel, restore_blas!, schema_json, set_path!, @cnlp_str

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
`CNLPModel("acopf"; ...)`, `@lib acopf`). Initialized from the colon-separated
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

    m = CNLPModel(cnlp"acopf"; args = grid)

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
"""
function load(path::AbstractString)
    blas_libs = [l.libname for l in BLAS.get_config().loaded_libs]
    handle = Libdl.dlopen(path, Libdl.RTLD_LOCAL | Libdl.RTLD_DEEPBIND)
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

The library's data schema as published by `<prefix>_schema` (ABI v2,
structured libraries only): a JSON description of the fields, kinds
(scalar/array/table) and column types.
"""
function schema_json(lib::CLib; prefix::AbstractString = "rec")
    fp = Libdl.dlsym(lib.handle, Symbol(prefix, "_schema"))
    n = ccall(fp, Cint, (Ptr{UInt8}, Cint), C_NULL, Cint(0))
    buf = Vector{UInt8}(undef, Int(n))
    ccall(fp, Cint, (Ptr{UInt8}, Cint), buf, n)
    return String(buf)
end

# Fill a builder from a NamedTuple: numbers are scalars, numeric vectors are
# arrays, vectors of named tuples are tables (sent column-by-column). The
# library itself validates names, kinds, completeness and column lengths.
function _fill_data(lib::CLib, prefix::AbstractString, data::NamedTuple)
    sym(s) = Symbol(prefix, "_", s)
    fp(s) = Libdl.dlsym(lib.handle, sym(s))
    b = ccall(fp(:data_begin), Cint, ())
    b > 0 || _status_error(sym(:data_begin), b)
    for (fname, val) in pairs(data)
        f = string(fname)
        if val isa AbstractFloat
            _check(ccall(fp(:set_scalar_f64), Cint, (Cint, Cstring, Cdouble),
                b, f, Cdouble(val)), sym(:set_scalar_f64))
        elseif val isa Integer
            _check(ccall(fp(:set_scalar_i64), Cint, (Cint, Cstring, Clonglong),
                b, f, Clonglong(val)), sym(:set_scalar_i64))
        elseif val isa AbstractVector && eltype(val) <: AbstractFloat
            v = convert(Vector{Float64}, val)
            _check(ccall(fp(:set_array_f64), Cint, (Cint, Cstring, Ptr{Cdouble}, Cint),
                b, f, v, Cint(length(v))), sym(:set_array_f64))
        elseif val isa AbstractVector && eltype(val) <: Integer
            v = convert(Vector{Int64}, val)
            _check(ccall(fp(:set_array_i64), Cint, (Cint, Cstring, Ptr{Clonglong}, Cint),
                b, f, v, Cint(length(v))), sym(:set_array_i64))
        elseif val isa AbstractVector && eltype(val) <: NamedTuple
            for (cn, ct) in zip(fieldnames(eltype(val)), fieldtypes(eltype(val)))
                c = string(cn)
                if ct <: AbstractFloat
                    col = Float64[getfield(r, cn) for r in val]
                    _check(ccall(fp(:set_col_f64), Cint,
                        (Cint, Cstring, Cstring, Ptr{Cdouble}, Cint),
                        b, f, c, col, Cint(length(col))), sym(:set_col_f64))
                elseif ct <: Integer
                    col = Int64[getfield(r, cn) for r in val]
                    _check(ccall(fp(:set_col_i64), Cint,
                        (Cint, Cstring, Cstring, Ptr{Clonglong}, Cint),
                        b, f, c, col, Cint(length(col))), sym(:set_col_i64))
                else
                    error("unsupported column type $ct for $f.$c")
                end
            end
        else
            error("unsupported field $f::$(typeof(val)): fields must be numbers, " *
                  "numeric vectors, or vectors of named tuples of numbers")
        end
    end
    ccall(fp(:data_ready), Cint, (Cint,), b) == 1 ||
        error("library reports data incomplete or inconsistent after all fields were set")
    id = ccall(fp(:new_from_data), Cint, (Cint,), b)
    id > 0 || _status_error(sym(:new_from_data), id)
    return id
end

"""
    CNLPModel(lib::CLib; args, prefix = "rec", name = basename(lib.path))

Create a model instance from `lib` and wrap it as an `AbstractNLPModel`.
`args` is what instantiates the model: an **integer** for simple libraries
(`<prefix>_new(n)` — one size knob), or a **named tuple** for structured
libraries (ABI v2 builder: numbers, numeric vectors, and vectors of named
tuples, sent columnar and validated against the library's schema). Any
number of instances may coexist per library. The first library call lazily
finishes its runtime initialization, after which the host's BLAS forwarding
is restored — see [`restore_blas!`](@ref).
"""
function CNLPModel(
    lib::CLib;
    args::Union{Integer, NamedTuple},
    prefix::AbstractString = "rec",
    name::AbstractString = basename(lib.path),
)
    sym(s) = Symbol(prefix, "_", s)
    fp(s) = Libdl.dlsym(lib.handle, sym(s))

    id = if args isa Integer
        i = ccall(fp(:new), Cint, (Cint,), Cint(args))
        i > 0 || _status_error(sym(:new), i)
        i
    else
        _fill_data(lib, prefix, args)
    end
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
    CNLPModel(name::AbstractString; kwargs...)

Name-based construction: resolve the library via [`lib`](@ref) and default
the symbol prefix to `name` (override with `prefix=`).

    set_path!("/opt/models")
    m = CNLPModel("acopf"; args = grid)
"""
CNLPModel(name::AbstractString; prefix::AbstractString = name, kwargs...) =
    CNLPModel(lib(name); prefix = prefix, kwargs...)

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
