orthogonal_random_features(nvertex::Int, dims...) =
    orthogonal_random_features(Float32, nvertex, dims...)

orthogonal_random_features(g, dims...) =
    orthogonal_random_features(float(eltype(g)), nv(g), dims...)

function orthogonal_random_features(::Type{T}, nvertex::Int, dims::Vararg{Int}) where {T}
    N = length(dims) + 2
    orf = Array{T,N}(undef, nvertex, nvertex, dims...)
    for cidx in CartesianIndices(dims)
        G = randn(nvertex, nvertex)
        A = qr(G)
        orf[:, :, cidx] .= collect(A.Q)
    end
    return orf
end

function laplacian_matrix(g, dims::Vararg{Int})
    L = laplacian_matrix(g)
    return repeat(L, outer=(1, 1, dims...))
end

node_identifier(g, dims...; method=orthogonal_random_features) = method(g, dims...)

function tokenize(g, node_feat::AbstractArray, edge_feat::AbstractArray; method=orthogonal_random_features)
    fg = FeaturedGraph(g)
    node_id = node_identifier(g, size(node_feat)[2:end]...; method=method)
    node_token = vcat(node_feat, node_id, node_id)
    el = to_namedtuple(fg)
    xs_id = NNlib.gather(node_id, batched_index(el.xs, size(node_id)[end]))
    nbr_id = NNlib.gather(node_id, batched_index(el.nbrs, size(node_id)[end]))
    edge_token = vcat(edge_feat, xs_id, nbr_id)
    return node_token, edge_token
end

function batched_index(idx::AbstractVector, batch_size::Integer)
    b = copyto!(similar(idx, 1, batch_size), collect(1:batch_size))
    return tuple.(idx, b)
end
