T = Float32

@testset "cuda/sparsematrix" begin
    adjm = cu(sparse(
        T[0 1 0 1;
          1 0 1 0;
          0 1 0 1;
          1 0 1 0]))
    @test collect(rowvals(adjm, 2)) == [1, 3]
    @test collect(nonzeros(adjm, 2)) == [1, 1]
end
