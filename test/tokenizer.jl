@testset "tokenizer" begin
    V = 4
    E = 5
    vdim = 3
    edim = 5
    batch_size = 10

    nf = rand(vdim, V)
    ef = rand(edim, E)

    adjm = [0 1 1 1;
            1 0 1 0;
            1 1 0 1;
            1 0 1 0]

    orthonormal = repeat(Matrix{Float64}(I(V)), outer=(1, 1, batch_size))

    node_id = node_identifier(adjm, batch_size; method=GraphSignals.orthogonal_random_features)
    @test NNlib.batched_mul(NNlib.batched_transpose(node_id), node_id) ≈ orthonormal

    node_id = node_identifier(adjm, batch_size; method=GraphSignals.laplacian_matrix)
    @test NNlib.batched_mul(NNlib.batched_transpose(node_id), node_id) ≈ orthonormal
end
