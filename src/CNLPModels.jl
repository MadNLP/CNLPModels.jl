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
using NLPModels
import NLPModels: increment!, @lencheck

export CNLPModel, restore_blas!

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
end

@noinline _status_error(sym, st) =
    error("$sym returned nonzero status $st")

@inline function _check(st::Cint, sym::Symbol)
    st == 0 || _status_error(sym, st)
    return nothing
end

"""
    CNLPModel(lib::CLib; n, prefix = "rec", name = basename(lib.path))

Create a new model instance of size `n` from `lib` (`<prefix>_new(n)`) and
wrap it as an `AbstractNLPModel`. Any number of instances may coexist per
library. The first library call lazily finishes its runtime initialization,
after which the host's BLAS forwarding is restored — see
[`restore_blas!`](@ref).
"""
function CNLPModel(
    lib::CLib;
    n::Integer,
    prefix::AbstractString = "rec",
    name::AbstractString = basename(lib.path),
)
    sym(s) = Symbol(prefix, "_", s)
    fp(s) = Libdl.dlsym(lib.handle, sym(s))

    id = ccall(fp(:new), Cint, (Cint,), Cint(n))
    id > 0 || _status_error(sym(:new), id)
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
    return CNLPModel(
        meta, Counters(), lib, id,
        fp(:obj), fp(:grad), fp(:cons),
        fp(:jac_structure), fp(:jac), fp(:hess_structure), fp(:hess),
    )
end

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
