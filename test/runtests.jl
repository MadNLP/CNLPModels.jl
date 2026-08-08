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
        @test_throws ErrorException CNLPModel(lib; prefix = "tq", n = 1)
    end

    m = CNLPModel(lib; prefix = "tq", n = 4)

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
        m6 = CNLPModel(lib; prefix = "tq", n = 6)
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
