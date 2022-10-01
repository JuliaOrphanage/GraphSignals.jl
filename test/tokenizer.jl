@testset "tokenizer" begin
    V, E = 4, 5
    batch_size = 10

    adjm = [0 1 1 1;
            1 0 1 0;
            1 1 0 1;
            1 0 1 0]

    orthonormal = repeat(Matrix{Float64}(I(V)), outer=(1, 1, batch_size))

    node_id = node_identifier(adjm, batch_size; method=GraphSignals.orthogonal_random_features)
    @test NNlib.batched_mul(NNlib.batched_transpose(node_id), node_id) ≈ orthonormal

    node_id = node_identifier(adjm, batch_size; method=GraphSignals.laplacian_matrix)
    @test NNlib.batched_mul(NNlib.batched_transpose(node_id), node_id) ≈ orthonormal

    node_id, edge_id = identifiers(adjm, batch_size)
    @test size(node_id) == (2V, V, batch_size)
    @test size(edge_id) == (2V, 2E, batch_size)
end
