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
    U = eigvecs(L)
    return repeat(U, outer=(1, 1, dims...))
end

"""
    node_identifier(g, dims...; method=GraphSignals.orthogonal_random_features)

Constructing node identifier for a graph `g` with additional dimensions `dims`.

# Arguments

- `g`: Data representing the graph topology. Possible type are
    - An adjacency matrix.
    - An adjacency list.
    - A Graphs' graph, i.e. `SimpleGraph`, `SimpleDiGraph` from Graphs, or `SimpleWeightedGraph`,
        `SimpleWeightedDiGraph` from SimpleWeightedGraphs.
    - An `AbstractFeaturedGraph` object.
- `dims`: Additional dimensions desired following after first two dimensions.
- `method`: Available methods are `GraphSignals.orthogonal_random_features` and
    `GraphSignals.laplacian_matrix`.

# Usage

```jldoctest
julia> using GraphSignals

julia> adjm = [0 1 1 1;
               1 0 1 0;
               1 1 0 1;
               1 0 1 0];

julia> batch_size = 10
10

julia> node_id = node_identifier(adjm, batch_size; method=GraphSignals.orthogonal_random_features);

julia> size(node_id)
(4, 4, 10)
```

See also [`tokenize`](@ref) for node/edge features tokenization.
"""
node_identifier(g, dims...; method=orthogonal_random_features) = method(g, dims...)

"""
    tokenize(g, node_feat, edge_feat; method=orthogonal_random_features)

Returns tokenized node features and edge features, respectively.

# Arguments

- `g`: Data representing the graph topology. Possible type are
    - An adjacency matrix.
    - An adjacency list.
    - A Graphs' graph, i.e. `SimpleGraph`, `SimpleDiGraph` from Graphs, or `SimpleWeightedGraph`,
        `SimpleWeightedDiGraph` from SimpleWeightedGraphs.
    - An `AbstractFeaturedGraph` object.
- `node_feat::AbstractArray`: Node features.
- `edge_feat::AbstractArray`: Edge features.
- `method`: Available methods are `GraphSignals.orthogonal_random_features` and
    `GraphSignals.laplacian_matrix`.

# Usage

```jldoctest
julia> using GraphSignals

```

See also [`node_identifier`](@ref) for generating node identifier.
"""
function tokenize(g, node_feat::AbstractArray, edge_feat::AbstractArray; method=orthogonal_random_features)
    fg = FeaturedGraph(g)
    node_id = node_identifier(g, size(node_feat)[3:end]...; method=method)
    node_token = vcat(node_feat, node_id, node_id)
    idx, nbrs, xs = collect(edges(fg))
    edge_feat = NNlib.gather(edge_feat, batched_index(idx, size(edge_feat)[end]))
    xs_id = NNlib.gather(node_id, batched_index(xs, size(node_id)[end]))
    nbr_id = NNlib.gather(node_id, batched_index(nbrs, size(node_id)[end]))
    edge_token = vcat(edge_feat, xs_id, nbr_id)
    return node_token, edge_token
end

function batched_index(idx::AbstractVector, batch_size::Integer)
    b = copyto!(similar(idx, 1, batch_size), collect(1:batch_size))
    return tuple.(idx, b)
end
