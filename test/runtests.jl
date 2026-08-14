using Test
using CNLPModels
using NLPModels
using LinearAlgebra
using MadNLP

# Build the hermetic C fixture (no Julia involved on the producer side).
const FIXDIR = joinpath(@__DIR__, "fixtures")
const LIBPATH = joinpath(mktempdir(), "libtinyqp.so")
cc = something(Sys.which("cc"), Sys.which("gcc"), Sys.which("clang"))
@test cc !== nothing
run(`$cc -shared -fPIC -O2 -o $LIBPATH $(joinpath(FIXDIR, "tinyqp.c"))`)

@testset "CNLPModels" begin
    lib = CNLPModels.load(LIBPATH)

    @testset "init failure surfaces" begin
        @test_throws ErrorException CNLPModel(lib, 1; prefix = "tq")
    end

    m = CNLPModel(lib, 4; prefix = "tq")

    @testset "meta" begin
        @test m.meta.nvar == 4
        @test m.meta.ncon == 1
        @test m.meta.nnzj == 2
        @test m.meta.nnzh == 4
        @test m.meta.x0 == zeros(4)
        @test all(==(-Inf), m.meta.lvar)
        @test all(==(Inf), m.meta.uvar)
        @test m.meta.lcon == [0.0]
        @test m.meta.ucon == [0.0]
    end

    @testset "evaluations match closed form" begin
        x = [0.5, 0.25, 2.0, -1.0]
        @test obj(m, x) == 0.25 + 0.5625 + 1.0 + 4.0
        @test grad(m, x) == 2.0 .* (x .- 1.0)
        @test cons(m, x) == [x[1] + x[2] - 1.0]
        @test jac_structure(m) == ([1, 1], [1, 2])
        @test jac_coord(m, x) == [1.0, 1.0]
        @test hess_structure(m) == ([1, 2, 3, 4], [1, 2, 3, 4])
        # y must not contribute (linear constraint); obj_weight must scale.
        @test hess_coord(m, x, [3.0]; obj_weight = 0.7) == fill(1.4, 4)
        @test hess_coord(m, x; obj_weight = 2.0) == fill(4.0, 4)
    end

    @testset "counters" begin
        @test m.counters.neval_obj > 0
        @test m.counters.neval_grad > 0
        @test m.counters.neval_hess > 0
    end

    @testset "BLAS guard is callable and BLAS works after it" begin
        @test restore_blas!(lib) === lib
        @test ones(2, 2) * ones(2, 2) == fill(2.0, 2, 2)
    end

    @testset "solve with MadNLP" begin
        res = madnlp(m; print_level = MadNLP.ERROR)
        @test res.status == MadNLP.SOLVE_SUCCEEDED
        @test res.objective ≈ 0.5 rtol = 1e-6
        @test res.solution[1] ≈ 0.5 rtol = 1e-6
        @test res.solution[2] ≈ 0.5 rtol = 1e-6
        @test all(res.solution[3:end] .≈ 1.0)
    end

    @testset "multiple coexisting instances" begin
        x = [0.5, 0.25, 2.0, -1.0]
        o_before = obj(m, x)
        m6 = CNLPModel(lib, 6; prefix = "tq")
        @test m6.meta.nvar == 6
        # Creating a second instance must not disturb the first (this was the
        # single-slot silent-corruption hazard of the pre-handle ABI).
        @test obj(m, x) == o_before
        @test m.id != m6.id
        res6 = madnlp(m6; print_level = MadNLP.ERROR)
        res4 = madnlp(m; print_level = MadNLP.ERROR)
        @test res6.status == MadNLP.SOLVE_SUCCEEDED
        @test res4.status == MadNLP.SOLVE_SUCCEEDED
        @test res6.objective ≈ 0.5 rtol = 1e-6
        @test res4.objective ≈ 0.5 rtol = 1e-6
    end
end

@testset "library path registry" begin
    # a second copy under a models dir, named like a registered model
    mdir = mktempdir()
    cp(LIBPATH, joinpath(mdir, "libtoyqp.so"))
    CNLPModels.set_path!(mdir)
    l = CNLPModels.lib("toyqp")
    @test l isa CNLPModels.CLib
    @test CNLPModels.lib("toyqp") === l          # cached by name
    m = CNLPModel("@toyqp", 4; prefix = "tq") # name-based construction
    @test m.meta.nvar == 4
    @test cnlp"toyqp" === l                      # the string-literal shortcut
    @test_throws ErrorException CNLPModels.lib("nonexistent")
end

@testset "arguments are positional, one per schema field" begin
    lib = CNLPModels.load(LIBPATH)
    # `tq` declares one scalar field and also exports tq_new, so a lone integer
    # takes the one-knob constructor.
    @test CNLPModels._schema_field_names(CNLPModels.schema_json(lib; prefix = "tq")) == ["n"]
    m5 = CNLPModel(lib, 5; prefix = "tq")
    @test m5.meta.nvar == 5
    r5 = madnlp(m5; print_level = MadNLP.ERROR)
    @test r5.status == MadNLP.SOLVE_SUCCEEDED
    @test r5.objective ≈ 0.5 rtol = 1e-6

    # `sq` is builder-only (no sq_new) and declares three fields, so the
    # arguments bind positionally in the order the schema publishes them.
    @test CNLPModels._schema_field_names(CNLPModels.schema_json(lib; prefix = "sq")) ==
          ["n", "s", "w"]
    n, s, w = 4, 2.0, [1.0, 2.0, 3.0, 4.0]
    ms = CNLPModel(lib, n, s, w; prefix = "sq")
    @test ms.meta.nvar == 4
    @test ms.meta.ncon == 1
    x = [0.5, 0.25, 2.0, -1.0]
    @test obj(ms, x) == sum(w .* (x .- s) .^ 2)          # min Σ wᵢ(xᵢ-s)²
    @test grad(ms, x) == 2.0 .* w .* (x .- s)
    @test hess_coord(ms, x, [3.0]; obj_weight = 0.7) == 1.4 .* w
    @test cons(ms, x) == [x[1] + x[2] - 1.0]

    # Order is load-bearing: the same values in the wrong order are refused by
    # the library's own setters (s is f64-only, w is the array field).
    @test_throws ErrorException CNLPModel(lib, s, n, w; prefix = "sq")

    # Wrong arity names the schema rather than reaching the builder.
    @test_throws ErrorException CNLPModel(lib, 4, 2.0; prefix = "sq")
    @test_throws ErrorException CNLPModel(lib, 5, 6; prefix = "tq")
    # No arguments at all is the no-instance-data case; both schemas want some.
    @test_throws ErrorException CNLPModel(lib; prefix = "tq")
    @test_throws ErrorException CNLPModel(lib; prefix = "sq")
    # A value that cannot cross the boundary is reported as such.
    @test_throws ErrorException CNLPModel(lib, "five"; prefix = "tq")
    # Inconsistent structured data is the library's own call to make.
    @test_throws ErrorException CNLPModel(lib, 4, 2.0, [1.0, 2.0]; prefix = "sq")
end

@testset "several models in one library, selected by name" begin
    lib = CNLPModels.load(LIBPATH)
    x = [0.5, 0.25, 2.0, -1.0]
    n, s, w = 4, 2.0, [1.0, 2.0, 3.0, 4.0]

    # The fixture carries four models in ONE shared library. Naming one selects
    # it, and the name is the symbol prefix its ABI functions are exported under.
    m = CNLPModel(lib, :tq, 4)
    @test m.meta.nvar == 4
    @test m.meta.name == "tq"                       # the model, not the file

    # A second, unrelated model out of the same library: different schema,
    # different instantiation surface (builder-only), different objective.
    ms = CNLPModel(lib, :sq, n, s, w)
    @test ms.lib === m.lib                          # one library, one runtime
    @test obj(m, x) == sum((x .- 1.0) .^ 2)
    @test obj(ms, x) == sum(w .* (x .- s) .^ 2)
    @test madnlp(ms; print_level = MadNLP.ERROR).status == MadNLP.SOLVE_SUCCEEDED

    # Instances of DIFFERENT models coexist as freely as instances of one: each
    # keeps its own ids, and neither disturbs the other.
    m6 = CNLPModel(lib, :tq, 6)
    @test m6.meta.nvar == 6
    @test obj(m, x) == sum((x .- 1.0) .^ 2)
    @test obj(ms, x) == sum(w .* (x .- s) .^ 2)

    # A model consuming no instance data still names itself.
    @test CNLPModel(lib, :fx).meta.name == "fx"

    # The same spelling through every library argument: path, search-path name,
    # string literal. The prefix no longer has to follow the file name.
    @test CNLPModel(LIBPATH, :tq, 4).meta.nvar == 4
    @test CNLPModel("@toyqp", :tq, 4).meta.nvar == 4
    @test CNLPModel(cnlp"toyqp", :tq, 4).meta.nvar == 4

    # The display name stays overridable, and defaults to the model.
    @test CNLPModel(lib, :tq, 4; name = "mine").meta.name == "mine"

    # Schemas are per model, addressed the same way.
    @test CNLPModels._schema_field_names(CNLPModels.schema_json(lib, :sq)) ==
          ["n", "s", "w"]
    @test CNLPModels._schema_field_names(CNLPModels.schema_json(lib, :tq)) == ["n"]

    # A name the library does not carry is reported as such, at the name —
    # not several calls later as a missing builder surface.
    @test_throws "carries no model named `nosuch`" CNLPModel(lib, :nosuch, 4)
    @test_throws "carries no model named `nosuch`" CNLPModel(lib, :nosuch)
    # Naming the model and passing `prefix=` are the same knob, so asking for
    # both is not a method rather than one silently winning.
    @test_throws MethodError CNLPModel(lib, :tq, 4; prefix = "sq")
end

@testset "cnlp literal is the shortcut" begin
    @test cnlp"toyqp" === CNLPModels.lib("toyqp")
    m = CNLPModel(cnlp"toyqp", 5; prefix = "tq")   # directly as the lib argument
    @test m.meta.nvar == 5
    @test_throws ErrorException cnlp"nonexistent"
end

@testset "a fixed library needs no arguments" begin
    lib = CNLPModels.load(LIBPATH)
    # `fx` declares zero instantiation arguments (fx_nargs() == 0), so no
    # arguments instantiate it directly through fx_new — whose integer is
    # part of the C signature and ignored.
    m0 = CNLPModel(lib; prefix = "fx")
    @test m0.meta.nvar == 3
    @test m0.meta.ncon == 1
    @test obj(m0, [0.5, 0.5, 1.0]) == 0.5
    # An explicit integer still works, and lands on the same fixed model.
    m1 = CNLPModel(lib, 999; prefix = "fx")
    @test m1.meta.nvar == 3
end

@testset "a string is @name on the search path, or a literal path" begin
    # `@name` resolves on the search path and defaults the prefix to the name.
    @test CNLPModels._is_name("@toyqp")
    @test CNLPModels._default_prefix("@toyqp") == "toyqp"

    # Anything else is a filesystem path, exactly as written. A full path to
    # the library file, with the prefix defaulting from the file name
    # (libtinyqp.so → tinyqp), overridable as always:
    @test !CNLPModels._is_name(LIBPATH)
    @test CNLPModels._default_prefix(LIBPATH) == "tinyqp"
    mp = CNLPModel(LIBPATH, 4; prefix = "tq")
    @test mp.meta.nvar == 4
    # The handle is cached by absolute path: one dlopen per library.
    @test CNLPModels._resolve_spec(LIBPATH) === CNLPModels._resolve_spec(LIBPATH)
    # A fixed model by path needs nothing beyond the path.
    mf = CNLPModel(LIBPATH; prefix = "fx")
    @test mf.meta.nvar == 3

    # A path to a bundle DIRECTORY finds the library inside it, and the
    # prefix defaults from the directory name.
    bdir = mktempdir()
    mkpath(joinpath(bdir, "toyqp2", "lib"))
    cp(LIBPATH, joinpath(bdir, "toyqp2", "lib", "libtoyqp2.so"))
    @test CNLPModels._default_prefix(joinpath(bdir, "toyqp2")) == "toyqp2"
    md = CNLPModel(joinpath(bdir, "toyqp2"), 4; prefix = "tq")
    @test md.meta.nvar == 4

    # A bare string without `@` is a file in the current directory — NOT a
    # search-path name.
    cd(bdir) do
        cp(LIBPATH, joinpath(bdir, "qp.so"))
        mc = CNLPModel("qp.so", 4; prefix = "tq")
        @test mc.meta.nvar == 4
        # `toyqp` is on the search path from the registry testset above, but
        # without `@` it is a missing local file, and says so.
        @test_throws "no shared library at" CNLPModel("toyqp", 4)
    end

    # A path that is not there fails as a path, never as a search-path miss.
    @test_throws "no shared library at" CNLPModel(joinpath(FIXDIR, "libnope.so"), 1)
end

@testset "search path initializes from the environment" begin
    old = copy(CNLPModels._PATHS)
    withenv("CNLPMODELS_PATH" => "/a:/b") do
        empty!(CNLPModels._PATHS)
        @test CNLPModels._paths() == ["/a", "/b"]
    end
    CNLPModels.set_path!(old...)
end

@testset "a directory without a library says what it tried" begin
    @test_throws "no shared library in" CNLPModel(mktempdir(), 1)
end

@testset "a table field binds column by column" begin
    lib = CNLPModels.load(LIBPATH)
    # min Σ_j w_j (x_{i_j} - 1)^2 over 3 vars: variable 1 carries rows 1 and 4.
    pts = [(i = 1, w = 1.0), (i = 2, w = 2.0), (i = 3, w = 3.0), (i = 1, w = 0.5)]
    m = CNLPModel(lib, pts; prefix = "tb")
    @test (m.meta.nvar, m.meta.ncon, m.meta.nnzj, m.meta.nnzh) == (3, 0, 0, 3)
    x = [0.0, 2.0, 4.0]
    @test obj(m, x) == 1.5 + 2.0 + 27.0
    @test grad(m, x) == [-3.0, 4.0, 18.0]
    hr, hc = hess_structure(m)
    @test hr == [1, 2, 3] && hc == [1, 2, 3]
    @test hess_coord(m, x, Float64[]; obj_weight = 0.5) == [1.5, 2.0, 3.0]
    # A column the library refuses surfaces as its setter's status.
    @test_throws "set_col_i64" CNLPModel(lib, [(i = 1, w = 1.0) for _ in 1:9]; prefix = "tb")
    # And a column type nothing can carry is refused before the boundary.
    @test_throws "unsupported column" CNLPModel(lib, [(i = 1, w = "x")]; prefix = "tb")
end

@testset "integer weights widen through the i64 array setter" begin
    lib = CNLPModels.load(LIBPATH)
    m = CNLPModel(lib, 4, 2.0, [1, 2, 3, 4]; prefix = "sq")
    x = [0.5, 0.25, 2.0, -1.0]
    @test obj(m, x) == sum([1.0, 2.0, 3.0, 4.0] .* (x .- 2.0) .^ 2)
    # A field value no ABI slot can carry is refused with its type named.
    @test_throws "unsupported field" CNLPModel(lib, 4, 2.0, "w"; prefix = "sq")
end

@testset "jprod and jtprod go through the cached structure" begin
    lib = CNLPModels.load(LIBPATH)
    m = CNLPModel(lib, 4; prefix = "tq")
    x = [0.5, 0.25, 2.0, -1.0]
    v = [1.0, 2.0, 3.0, 4.0]
    y = [5.0]
    J = zeros(m.meta.ncon, m.meta.nvar)
    rows, cols = jac_structure(m)
    vals = jac_coord(m, x)
    for k in eachindex(vals)
        J[rows[k], cols[k]] += vals[k]
    end
    @test jprod(m, x, v) ≈ J * v
    @test jtprod(m, x, y) ≈ J' * y
    @test NLPModels.jprod_nln!(m, x, v, zeros(1)) ≈ J * v
    @test NLPModels.jtprod_nln!(m, x, y, zeros(4)) ≈ J' * y
end

@testset "unbundled juliac libraries are detected by their libjulia NEEDED" begin
    # A plain C library links no libjulia: it must take the ordinary dlopen
    # path (the whole suite above ran through it).
    @test CNLPModels._stdlinked_libjulia(LIBPATH) === nothing
    # Unreadable / not-an-object paths are left for dlopen to complain about.
    @test CNLPModels._stdlinked_libjulia("/nonexistent/libfoo.so") === nothing
    @test CNLPModels._stdlinked_libjulia(joinpath(FIXDIR, "tinyqp.c")) === nothing

    # A library that links the standard libjulia soname is detected, with its
    # version — the trigger for the private-runtime route. Built here by
    # linking the fixture against this process's own libjulia. (Detection is
    # a whole-function return, not a loop-local one: a regression to the
    # latter loses the value and silently downgrades to the aborting path.)
    if Sys.islinux()
        using Libdl
        libjulia = realpath(Libdl.dlpath("libjulia"))
        ver = "$(VERSION.major).$(VERSION.minor)"
        linked = joinpath(mktempdir(), "libtinyqp_jl.so")
        run(`$cc -shared -fPIC -O2 -o $linked $(joinpath(FIXDIR, "tinyqp.c"))
             -Wl,--no-as-needed $libjulia`)
        @test CNLPModels._stdlinked_libjulia(linked) == ver
    end
end

@testset "named blocks" begin
    lib = CNLPModels.load(LIBPATH)
    # The fixture publishes a layout: `x` (variable), `budget` (constraint),
    # `w` (parameter). A consumer handed only the library learns the names the
    # model was written with, and addresses slices by them.
    m = CNLPModel(lib, 4; prefix = "tq")
    @test keys(get_vars(m)) == (:x,)
    @test keys(get_cons(m)) == (:budget,)
    @test keys(get_pars(m)) == (:w,)

    b = get_vars(m, :x)
    @test b === get_vars(m).x
    @test (b.kind, b.offset, b.length, b.dims) == (:var, 0, 4, (4,))
    # Lengths follow the INSTANCE, not the library: a second model at another
    # size reports its own.
    @test get_vars(CNLPModel(lib, 7; prefix = "tq"), :x).length == 7

    # The two ways of asking wrongly are distinguished: a name of the wrong
    # kind names the accessor that would work, a typo lists what exists.
    @test_throws "is a parameter (get_pars), not a var" get_vars(m, :w)
    @test_throws "is a variable (get_vars), not a par" get_pars(m, :x)
    @test_throws "no named variable `nope`" get_vars(m, :nope)

    # Result extraction, against a hand-built result: the block reshapes its
    # own slice, which is the whole point of publishing dims.
    res = (solution = [1.0, 2.0, 3.0, 4.0], multipliers = [7.0],
           multipliers_L = [0.1, 0.2, 0.3, 0.4], multipliers_U = [0.5, 0.6, 0.7, 0.8])
    @test solution(res, b) == [1.0, 2.0, 3.0, 4.0]
    @test multipliers(res, get_cons(m, :budget)) == [7.0]
    @test multipliers_L(res, b) == [0.1, 0.2, 0.3, 0.4]
    @test multipliers_U(res, b) == [0.5, 0.6, 0.7, 0.8]

    # Parameters are live state, per instance.
    p = get_pars(m, :w)
    set_value!(m, p, [3.0, 4.0])
    @test get_value(m, p) == [3.0, 4.0]
    m2 = CNLPModel(lib, 4; prefix = "tq")
    set_value!(m2, get_pars(m2, :w), [9.0, 9.0])
    @test get_value(m, p) == [3.0, 4.0]          # the first is undisturbed
    @test_throws DimensionMismatch set_value!(m, p, [1.0])
    @test_throws "not a parameter" get_value(m, b)

    # Named blocks are an optional ABI surface, not a requirement: `sq`
    # publishes none — as a hand-written C library may — and everything else
    # about it still works.
    ms = CNLPModel(lib, 4, 2.0, [1.0, 2.0, 3.0, 4.0]; prefix = "sq")
    @test keys(get_vars(ms)) == ()
    @test_throws "publishes no named blocks" get_vars(ms, :x)
    @test ms.meta.nvar == 4
end
