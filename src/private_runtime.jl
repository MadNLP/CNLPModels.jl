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
# Linux and macOS. The macOS half follows the same shape through
# `install_name_tool`: salted basenames become salted install-name ids
# (`@rpath/<salt>_libjulia.x.y.dylib`), load commands are rewritten in place
# of NEEDED entries, LC_RPATHs point at the scratch layout, and the edited
# copies are re-signed ad hoc (mandatory on arm64 — an edited, unsigned
# binary is killed at load). No ELF symbol-version stamp exists or is
# needed: Mach-O two-level namespace binds each symbol to the install name
# it was found under, so distinctly-named runtimes cannot cross-bind.
# Measured before this was written (2026-08-13, on the target machine): a
# JuliaC-privatized bundle — the same transformation done at compile time —
# loads into a running Julia on macOS and evaluates correctly, while the
# raw unbundled form aborts exactly as on Linux.
#
# Windows has no privatization at all; there the unbundled form is refused
# with an explanation — strictly better than the abort — and the bundled
# form works everywhere.

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

# The platform-form basenames of the runtime pair a library of version `ver`
# links: what the copies are named (salted), and what is matched when
# rewriting references.
_rt_basenames(ver) =
    Sys.isapple() ? ("libjulia." * ver * ".dylib", "libjulia-internal." * ver * ".dylib") :
    ("libjulia.so." * ver, "libjulia-internal.so." * ver)

function _dlopen_private(path::AbstractString, ver::AbstractString, flags)
    hostver = "$(VERSION.major).$(VERSION.minor)"
    ver == hostver || error(
        "$path is an unbundled juliac library linked against Julia $ver, but " *
        "this process runs Julia $hostver. Loading it here needs a runtime " *
        "sourced from a matching installation — run under Julia $ver, or " *
        "recompile the library (compile_library(...; bundle = true) makes it " *
        "self-contained).")
    (Sys.islinux() || Sys.isapple()) || error(
        "$path is an unbundled juliac library, which this Julia process can " *
        "only load on Linux or macOS (where CNLPModels can give it a private " *
        "copy of the installed runtime; loaded as-is it would abort the " *
        "process). Compile with bundle = true for this platform, or consume " *
        "the library from Python/C, where it works as-is.")

    libdir = dirname(Libdl.dlpath("libjulia"))     # <prefix>/lib
    # The scratch runtime is removed at process exit. CNLPMODELS_KEEP_RUNTIME=1
    # keeps it, so the patched binaries can be inspected after the fact —
    # a successful load otherwise cleans up the very artifacts that show HOW
    # it succeeded (a failed one leaks them, which is the misleading half:
    # the only dir left to inspect is then the broken round's).
    keep = get(ENV, "CNLPMODELS_KEEP_RUNTIME", "") == "1"
    scratch = mktempdir(prefix = "cnlpmodels_rt_", cleanup = !keep)
    lib = joinpath(scratch, "lib");     mkpath(lib)
    priv = joinpath(lib, "julia");      mkpath(priv)

    # Entropy straight from the OS: task-local RNG state has produced repeated
    # salts across sequential calls, and equal salts mean one shared — and
    # aborting — runtime.
    salt = String(rand(Random.RandomDevice(), ['a':'z'; 'A':'Z'; '0':'9'], 8))

    # Salted runtime pair, under platform-form names: that is the form the
    # dep_libs references resolve, and the form the model's references are
    # rewritten to.
    #
    # The naming scheme differs per platform, and the difference is
    # load-bearing. Linux prepends the salt (`<salt>_libjulia.so.x.y`) —
    # patchelf grows ELF strings freely. On macOS every rewrite must fit the
    # string it replaces (a juliac dylib has no Mach-O header padding to grow
    # into — measured on the target, round 3), so the salt REPLACES the
    # 8-char stem: `libjulia` → `lib<salt5>`, keeping every name, id,
    # load-command string and dep_libs entry exactly the same length
    # (`libjulia-internal.dylib` → `libXXXXX-internal.dylib`). References to
    # the runtime appear in versioned AND unversioned forms; symlinks below
    # cover the alias forms, as the real installation's symlink chain does.
    base_lj, base_int = _rt_basenames(ver)
    src_lj  = joinpath(libdir, base_lj)
    src_int = joinpath(libdir, "julia", base_int)
    (isfile(src_lj) && isfile(src_int)) || error(
        "this Julia installation has no $base_lj/$base_int pair at $libdir " *
        "to source a private runtime from")
    if Sys.islinux()
        s_lj  = joinpath(lib,  salt * "_" * base_lj)
        s_int = joinpath(priv, salt * "_" * base_int)
    else
        stem = "lib" * lowercase(salt[1:5])
        stem == "libjulia" && (stem = "libjulja")   # never collide with the host stem
        s_lj  = joinpath(lib,  _restem(base_lj, stem))
        s_int = joinpath(priv, _restem(base_int, stem))
    end
    for (src, dst) in ((src_lj, s_lj), (src_int, s_int))
        cp(realpath(src), dst)
        chmod(dst, 0o755)
    end

    if Sys.islinux()
        rpath = lib * ":" * priv
        oldver = "JL_LIBJULIA_" * hostver
        newver = "JL_" * salt * "_" * hostver
        for dst in (s_lj, s_int)
            _patchelf("--set-soname", basename(dst), dst)
            _patchelf("--set-rpath", rpath, dst)
        end
        _replace_dep_libs(s_lj, "libjulia" => salt * "_libjulia")
        for dst in (s_lj, s_int)
            _rewrite_needed(dst, ver, s_lj, s_int)
            PatchVersion.patch_version!(dst, oldver, newver)
        end
    else
        # macOS: identity is the install name — salting it is the whole of
        # what the ELF path needed sonames AND symbol versions for (two-level
        # namespace binds symbols to the install name they were found under).
        # No rpath is ADDED anywhere and no string GROWS anywhere: adds need
        # header padding these binaries lack (measured, round 3), and a grown
        # `-change` is the same wall (untested, not risked — round 4). The
        # stem substitution keeps every rewrite length-identical. The salted
        # pair needs no rpath edits — its `@loader_path`-relative rpaths
        # resolve inside the scratch mirror of the real layout. Every binary
        # edit happens before the ad-hoc re-sign below.
        for dst in (s_lj, s_int)
            _install_name_tool("-id", "@rpath/" * basename(dst), dst)
        end
        _replace_dep_libs(s_lj, "libjulia" => stem)
        for dst in (s_lj, s_int)
            _rewrite_loads_macos(dst, stem)
            _codesign_adhoc(dst)
        end
        # References use versioned and unversioned forms interchangeably —
        # cover the aliases with symlinks, as the real installation's chain
        # does (`libjulia-internal.dylib` → `.1.12.dylib` → `.1.12.6.dylib`).
        for (alias, dst) in (
            (stem * ".dylib", s_lj),
            (_restem(basename(realpath(src_lj)), stem), s_lj),
            (stem * "-internal.dylib", s_int),
            (_restem(basename(realpath(src_int)), stem), s_int),
        )
            p = joinpath(dirname(dst), alias)
            alias == basename(dst) || ispath(p) || symlink(basename(dst), p)
        end
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
    if Sys.islinux()
        rpath = lib * ":" * priv
        _rewrite_needed(model, ver, s_lj, s_int)
        _patchelf("--set-rpath", rpath, model)
        PatchVersion.patch_version!(model, "JL_LIBJULIA_" * hostver,
                                    "JL_" * salt * "_" * hostver)
    else
        _rewrite_loads_macos(model, stem)
        _retarget_rpaths_macos(model, lib, priv)
        _codesign_adhoc(model)
    end

    return Libdl.dlopen(model, flags)
end

# ── macOS binary edits ───────────────────────────────────────────────────────

# The tool's own stderr is the diagnostic (it names padding problems
# precisely), so a failure carries it — discarding it cost a debugging
# round on the target machine.
function _install_name_tool(args...)
    tool = Sys.which("install_name_tool")
    tool === nothing && error(
        "install_name_tool not found — the Xcode command-line tools are " *
        "needed to load an unbundled juliac library on macOS")
    cmd = `$tool $(collect(String, args))`
    buf = IOBuffer()
    success(pipeline(cmd; stdout = buf, stderr = buf)) ||
        error("install_name_tool failed: ", strip(String(take!(buf))),
              "\n(command: ", cmd, ")")
    return nothing
end

# `libjulia<rest>` → `<stem><rest>`, length-identical by construction.
function _restem(name::AbstractString, stem::AbstractString)
    startswith(name, "libjulia") ||
        error("$name does not carry the libjulia stem to substitute")
    return stem * name[9:end]
end

# Rewrite whatever load commands reference the runtime onto the salted stem.
# The OLD string must be exactly as recorded in the binary — references
# appear versioned (`@rpath/libjulia.x.y.dylib`) and unversioned
# (`@rpath/libjulia-internal.dylib`), so they are read out with ObjectFile
# and matched by stem, never assumed. An `@`-relative reference keeps its
# prefix (same length); an absolute one becomes `@rpath/` + name (strictly
# shorter). Nothing ever grows.
function _rewrite_loads_macos(bin::AbstractString, stem::AbstractString)
    loads = open(bin, "r") do io
        oh = only(ObjectFile.readmeta(io))
        [ObjectFile.path(dl) for dl in ObjectFile.DynamicLinks(oh)]
    end
    for p in loads
        b = basename(p)
        occursin(r"^libjulia(-internal)?(\.\d+)*\.dylib$", b) || continue
        prefix = startswith(p, "@") ? p[1:(end - length(b))] : "@rpath/"
        _install_name_tool("-change", p, prefix * _restem(b, stem), bin)
    end
    return nothing
end

# The existing LC_RPATH entries, exactly as recorded. `otool` ships beside
# `install_name_tool`, so its absence is caught by the same error there.
function _rpaths_macos(bin::AbstractString)
    tool = Sys.which("otool")
    tool === nothing && return String[]
    out = read(`$tool -l $bin`, String)
    paths = String[]
    lines = split(out, '\n')
    for (i, l) in enumerate(lines)
        occursin(r"\bcmd LC_RPATH\b", l) || continue
        for j in (i + 1):min(i + 3, length(lines))
            m = match(r"^\s*path (.*?) \(offset \d+\)\s*$", lines[j])
            m === nothing || (push!(paths, String(m.captures[1])); break)
        end
    end
    return paths
end

# An unbundled juliac library carries absolute rpaths into the Julia
# installation it was linked against (JuliaC's RPATH_JULIA: `<prefix>/lib`
# and `<prefix>/lib/julia`). Retarget them IN PLACE onto the scratch layout —
# `install_name_tool -rpath old new` rewrites within the existing load
# command, so unlike `-add_rpath` it needs no header padding as long as the
# new path is no longer than the recorded one; scratch paths are short.
function _retarget_rpaths_macos(bin::AbstractString, lib, priv)
    existing = _rpaths_macos(bin)
    isempty(existing) && error(
        "$bin carries no LC_RPATH to retarget — cannot point it at a " *
        "private runtime")
    hit = false
    for p in unique(existing)
        new = endswith(rstrip(p, '/'), "/julia") ? priv : lib
        p == new && continue
        _install_name_tool("-rpath", p, new, bin)
        hit = true
    end
    hit || error("no rpath of $bin could be retargeted onto $lib")
    return nothing
end

# Mandatory on arm64: install_name_tool invalidates the signature, and an
# invalidly-signed binary is killed at load, not merely warned about. Ad-hoc
# (`-s -`) is exactly what JuliaC's bundler applies. A failure surfaces its
# own text; a missing codesign is skipped (Intel tolerates it, and on arm64
# the subsequent dlopen failure names the binary).
function _codesign_adhoc(bin::AbstractString)
    cs = Sys.which("codesign")
    cs === nothing && return nothing
    cmd = `$cs -f -s - $bin`
    buf = IOBuffer()
    success(pipeline(cmd; stdout = buf, stderr = buf)) ||
        error("codesign failed: ", strip(String(take!(buf))),
              "\n(command: ", cmd, ")")
    return nothing
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

function _replace_dep_libs(file::AbstractString, subst::Pair{<:AbstractString, <:AbstractString})
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
        patched = Vector{UInt8}(replace(blob, subst)[1:_DEP_LIBS_LENGTH])
        data[off:(off + _DEP_LIBS_LENGTH - 1)] .= patched
        Mmap.sync!(data)
    end
    return nothing
end
