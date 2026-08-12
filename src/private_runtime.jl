# ── Loading UNBUNDLED juliac libraries in-process ────────────────────────────
#
# A `juliac --output-lib` library compiled WITHOUT bundling links the standard
# `libjulia.so.x.y` and embeds a full (trimmed) system image. Its entry-point
# preamble initializes that runtime lazily on the first call
# (`jl_autoinit_and_adopt_thread`). Loaded into a process that is already
# Julia, the shared `libjulia` is already initialized, so the preamble falls
# through to `jl_adopt_thread`, whose `jl_init_threadtls` guard
# (`if (jl_get_pgcstack() != NULL) abort();`) kills the whole process — and
# skipping the adoption would not help, because the library's image can only
# be restored by initializing a runtime with it. The library needs a SECOND
# runtime, distinct from the host's.
#
# JuliaC's bundler solves this at compile time by shipping a privatized
# runtime copy (~80 MB per model). This file solves it at LOAD time from the
# consumer's own Julia installation, so the shipped artifact stays one small
# file. The transformation is the bundler's, replayed in scratch:
#
#   1. copy `libjulia.so.x.y` and `julia/libjulia-internal.so.x.y` under
#      salted soname-form names (`<salt>_libjulia.so.x.y`), `--set-soname`;
#   2. rewrite libjulia's `dep_libs` blob (how it locates libjulia-internal
#      at init) to the salted names;
#   3. symlink the rest of `lib/julia` beside them (support libraries keep
#      their standard sonames — in-process they dedupe with the host's
#      already-loaded copies, exactly as a bundle's do);
#   4. re-stamp the ELF symbol version `JL_LIBJULIA_x.y` → `JL_<salt>_x.y` in
#      both copies, so nothing binds across to the host's libjulia;
#   5. patch a scratch copy of the model library — NEEDED entries onto the
#      salted names, rpath onto the scratch layout, same version stamp — and
#      `dlopen` that copy.
#
# The first entry-point call then initializes the salted runtime with the
# model's image and adopts the calling thread there, legitimately: that
# runtime's thread-local state is fresh. This is byte-for-byte the topology
# of a loaded bundle (which also shares every non-julia dependency with the
# host by soname), just sourced locally, which is why the same libblastrampoline
# caveat and `restore_blas!` seam apply.
#
# The salt is fresh per load: two identically-named runtimes cannot coexist
# in one process (the loader would satisfy the second library's NEEDED with
# the first's runtime, resurrecting the abort), while distinctly-salted ones
# coexist fine. Loads are cached per file, so one library costs one runtime.
#
# Linux-only: privatization on macOS would follow the same shape via
# install_name_tool (JuliaC implements the compile-time half), but the
# load-time half is not written yet; Windows has no privatization at all.
# On those platforms an unbundled library is refused with an explanation —
# strictly better than the abort — and the bundled form works everywhere.

import .PatchVersion
import ObjectFile
import Patchelf_jll
import Random
import Mmap

# Model-library handles by realpath, so loading the same file twice reuses
# one private runtime instead of standing up a second.
const _PRIVATE_HANDLES = Dict{String, Ptr{Cvoid}}()

_patchelf(args...) =
    run(`$(Patchelf_jll.patchelf()) $(collect(String, args))`)

# The standard (unprivatized) libjulia soname a library links, or `nothing`.
# A bundled library links a salted name (`<salt>_libjulia.so.x.y`), so it
# does not match; neither does a plain C library, or anything readmeta cannot
# parse (which is then left to dlopen to judge).
function _stdlinked_libjulia(path::AbstractString)
    occursin(r"\.(so|dylib)", basename(path)) || return nothing
    return try
        open(path, "r") do io
            oh = only(ObjectFile.readmeta(io))
            for dl in ObjectFile.DynamicLinks(oh)
                m = match(r"^libjulia\.(?:so\.(\d+\.\d+)|(\d+\.\d+)\.dylib)$",
                          basename(ObjectFile.path(dl)))
                m === nothing || return something(m.captures...)
            end
            return nothing
        end
    catch
        nothing
    end
end

# dlopen for model libraries: unbundled juliac libraries take the private-
# runtime route; everything else (bundles, plain C libraries) loads as-is.
function _dlopen_model(path::AbstractString, flags)
    ver = _stdlinked_libjulia(path)
    ver === nothing && return Libdl.dlopen(path, flags)
    key = realpath(path)
    return get!(_PRIVATE_HANDLES, key) do
        _dlopen_private(key, ver, flags)
    end
end

function _dlopen_private(path::AbstractString, ver::AbstractString, flags)
    hostver = "$(VERSION.major).$(VERSION.minor)"
    ver == hostver || error(
        "$path is an unbundled juliac library linked against Julia $ver, but " *
        "this process runs Julia $hostver. Loading it here needs a runtime " *
        "sourced from a matching installation — run under Julia $ver, or " *
        "recompile the library (compile_library(...; bundle = true) makes it " *
        "self-contained).")
    Sys.islinux() || error(
        "$path is an unbundled juliac library, which this Julia process can " *
        "only load on Linux (where CNLPModels can give it a private copy of " *
        "the installed runtime; loaded as-is it would abort the process). " *
        "Compile with bundle = true for this platform, or consume the " *
        "library from Python/C, where it works as-is.")

    libdir = dirname(Libdl.dlpath("libjulia"))     # <prefix>/lib
    scratch = mktempdir(prefix = "cnlpmodels_rt_")
    lib = joinpath(scratch, "lib");     mkpath(lib)
    priv = joinpath(lib, "julia");      mkpath(priv)
    rpath = lib * ":" * priv

    # Entropy straight from the OS: task-local RNG state has produced repeated
    # salts across sequential calls, and equal salts mean one shared — and
    # aborting — runtime.
    salt = String(rand(Random.RandomDevice(), ['a':'z'; 'A':'Z'; '0':'9'], 8))
    oldver = "JL_LIBJULIA_" * hostver
    newver = "JL_" * salt * "_" * hostver

    # Salted runtime pair, under soname-form names: that is the form the
    # dep_libs references resolve, and the form the model's NEEDED entries
    # are rewritten to.
    src_lj  = joinpath(libdir, "libjulia.so." * ver)
    src_int = joinpath(libdir, "julia", "libjulia-internal.so." * ver)
    (isfile(src_lj) && isfile(src_int)) || error(
        "this Julia installation has no $(basename(src_lj))/" *
        "$(basename(src_int)) pair at $libdir to source a private runtime from")
    s_lj  = joinpath(lib,  salt * "_libjulia.so." * ver)
    s_int = joinpath(priv, salt * "_libjulia-internal.so." * ver)
    for (src, dst) in ((src_lj, s_lj), (src_int, s_int))
        cp(realpath(src), dst)
        chmod(dst, 0o755)
        _patchelf("--set-soname", basename(dst), dst)
        _patchelf("--set-rpath", rpath, dst)
    end
    _replace_dep_libs(s_lj, salt)
    for dst in (s_lj, s_int)
        _rewrite_needed(dst, ver, s_lj, s_int)
        PatchVersion.patch_version!(dst, oldver, newver)
    end

    # The support libraries, found relative to the salted runtime exactly as
    # they are relative to the real one.
    for f in readdir(joinpath(libdir, "julia"))
        startswith(f, "libjulia") && continue
        symlink(joinpath(libdir, "julia", f), joinpath(priv, f))
    end

    # The model itself, onto the salted runtime.
    model = joinpath(lib, basename(path))
    cp(path, model)
    chmod(model, 0o755)
    _rewrite_needed(model, ver, s_lj, s_int)
    _patchelf("--set-rpath", rpath, model)
    PatchVersion.patch_version!(model, oldver, newver)

    return Libdl.dlopen(model, flags)
end

function _rewrite_needed(bin::AbstractString, ver, s_lj, s_int)
    needed = filter(!isempty, split(
        read(`$(Patchelf_jll.patchelf()) --print-needed $bin`, String), '\n'))
    for n in needed
        if n == "libjulia.so." * ver
            _patchelf("--replace-needed", n, basename(s_lj), bin)
        elseif n == "libjulia-internal.so." * ver
            _patchelf("--replace-needed", n, basename(s_int), bin)
        end
    end
end

# libjulia locates libjulia-internal (and codegen, when present) through its
# embedded `dep_libs` path list, not through NEEDED — rewrite those
# references onto the salted names. Same transformation, buffer cap and
# rationale as JuliaC's bundler (`replace_dep_libs`).
const _DEP_LIBS_LENGTH = 512

function _replace_dep_libs(file::AbstractString, salt::AbstractString)
    off = open(file, "r") do io
        obj = only(ObjectFile.readmeta(io))
        syms = collect(ObjectFile.Symbols(obj))
        i = findfirst(==(ObjectFile.mangle_symbol_name(obj, "dep_libs")),
                      ObjectFile.symbol_name.(syms))
        i === nothing && error(
            "no dep_libs symbol in $file — cannot privatize this libjulia")
        ObjectFile.symbol_offset(syms[i])
    end
    open(file, "r+") do io
        data = Mmap.mmap(io)
        blob = String(data[off:(off + _DEP_LIBS_LENGTH - 1)])
        patched = Vector{UInt8}(
            replace(blob, "libjulia" => salt * "_libjulia")[1:_DEP_LIBS_LENGTH])
        data[off:(off + _DEP_LIBS_LENGTH - 1)] .= patched
        Mmap.sync!(data)
    end
    return nothing
end
