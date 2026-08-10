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
    m = CNLPModel("toyqp", 4; prefix = "tq") # name-based construction
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

@testset "cnlp literal is the shortcut" begin
    @test cnlp"toyqp" === CNLPModels.lib("toyqp")
    m = CNLPModel(cnlp"toyqp", 5; prefix = "tq")   # directly as the lib argument
    @test m.meta.nvar == 5
    @test_throws ErrorException cnlp"nonexistent"
end
