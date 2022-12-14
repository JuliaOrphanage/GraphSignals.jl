orthogonal_random_features(nvertex::Int, dims::Vararg{Int}) =
    orthogonal_random_features(Float32, nvertex, dims...)

orthogonal_random_features(::Type{T}, g, dims::Vararg{Int}) where {T} =
    orthogonal_random_features(T, nv(g), dims...)

orthogonal_random_features(g, dims::Vararg{Int}) =
    orthogonal_random_features(float(eltype(g)), nv(g), dims...)

function orthogonal_random_features(::Type{T}, nvertex::Int, dims::Vararg{Int}) where {T}
    N = length(dims) + 2
    orf = Array{T,N}(undef, nvertex, nvertex, dims...)
    for cidx in CartesianIndices(dims)
        G = randn(nvertex, nvertex)
        A = qr(G)
        copyto!(view(orf, :, :, cidx), A.Q)
    end
    return orf
end

function laplacian_matrix(::Type{T}, g, dims::Vararg{Int}) where {T}
    L = laplacian_matrix(g, T)
    U = eigvecs(L)
    return repeat(U, outer=(1, 1, dims...))
end

laplacian_matrix(g, dims::Vararg{Int}) = laplacian_matrix(float(eltype(g)), g, dims...)

"""
    node_identifier([T], g, dims...; method=GraphSignals.orthogonal_random_features)

Constructing node identifier for a graph `g` with additional dimensions `dims`.

# Arguments

- `T`: Element type of returning objects.
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

See also [`identifiers`](@ref) for node/edge identifiers.
"""
node_identifier(::Type{T}, g, dims...; method=orthogonal_random_features) where {T} =
    method(T, g, dims...)

node_identifier(g, dims...; method=orthogonal_random_features) =
    method(float(eltype(g)), g, dims...)

"""
    identifiers([T], g, dims...; method=orthogonal_random_features)

Returns node identifier and edge identifier.

# Arguments

- `T`: Element type of returning objects.
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

julia> V, E = 4, 5
(4, 5)

julia> batch_size = 10
10

julia> adjm = [0 1 1 1;
               1 0 1 0;
               1 1 0 1;
               1 0 1 0];

julia> node_id, edge_token = identifiers(adjm, batch_size);

julia> size(node_id)
(8, 4, 10)

julia> size(edge_id)
(8, 10, 10)
```

See also [`node_identifier`](@ref) for generating node identifier only.
"""
function identifiers(::Type{T}, g, dims...; method=orthogonal_random_features) where {T}
    fg = FeaturedGraph(g)
    node_id = node_identifier(T, g, dims...; method=method)
    node_token = vcat(node_id, node_id)
    el = to_namedtuple(fg)
    xs_id = NNlib.gather(node_id, batched_index(el.xs, size(node_id)[end]))
    nbr_id = NNlib.gather(node_id, batched_index(el.nbrs, size(node_id)[end]))
    edge_token = vcat(xs_id, nbr_id)
    return node_token, edge_token
end

identifiers(g, dims...; method=orthogonal_random_features) =
    identifiers(float(eltype(g)), g, dims...; method=method)

function batched_index(idx::AbstractVector, batch_size::Integer)
    b = copyto!(similar(idx, 1, batch_size), collect(1:batch_size))
    return tuple.(idx, b)
end

Base.depwarn("""tokenize() is removed from GraphSignals 0.8.2.
        It should be replaced with `identifiers`.""", :tokenize)
