"""
A indexing structure for accessing neighbors of a vertex. 
"""
struct EdgeIndex{T<:AbstractVector{<:AbstractVector}}
    iadjl::T
end

# make it support `iterate` to be a iterator
function EdgeIndex(iadjl::AbstractVector{<:Vector{T}}) where {T<:Integer}
    a = convert(Vector{Vector{Tuple{Int64, Int64}}}, iadjl)
    EdgeIndex{typeof(a)}(a)
end

function EdgeIndex(g)
    iadjl = order_edges(adjacency_list(g), directed=is_directed(g))
    EdgeIndex(iadjl)
end

nv(ei::EdgeIndex) = length(ei.iadjl)

ne(ei::EdgeIndex) = length(unique(map(x -> x[2], vcat(Array.(ei.iadjl)...))))

neighbors(ei::EdgeIndex, i) = ei.iadjl[i]

get(ei::EdgeIndex, key::NTuple{2}, default=nothing) = get(ei, key..., default)
get(ei::EdgeIndex, key::CartesianIndex{2}, default=nothing) = get(ei, key[1], key[2], default)

function get(ei::EdgeIndex, i, j, default=nothing)
    nbs = neighbors(ei, i)
    return _get(nbs, j, default)
end

function _get(x::AbstractVector, j, default)
    # linear search (O(n)) here can be optimized with hash table (O(1))
    for (x_j, v) in x
        x_j == j && return v
    end
    return default
end

"""
Order the edges in a graph by giving a unique integer to each edge.
"""
function order_edges(adjl::AbstractVector{<:AbstractVector}; directed::Bool=false)
    T = Vector{Tuple{Int64, Int64}}
    res = similar(adjl, T)
    for i = 1:length(res)
        res[i] = T[]
    end
    if directed
        directed_order_edges!(res, adjl)
    else
        undirected_order_edges!(res, adjl)
    end
    return res
end

function undirected_order_edges!(res, adjl::AbstractVector{<:AbstractVector})
    viewed = Set{Tuple{Int64, Int64}}()
    k = 1
    for i = 1:length(adjl)
        for j = adjl[i]
            if i == j
                push!(res[i], (j, k))
                k += 1
            elseif !((i, j) in viewed)
                push!(res[i], (j, k))
                push!(res[j], (i, k))
                push!(viewed, (j, i))
                k += 1
            end
        end
    end
    res
end

function directed_order_edges!(res, adjl::AbstractVector{<:AbstractVector})
    k = 1
    for i = 1:length(adjl)
        for j = adjl[i]
            push!(res[i], (j, k))
            k += 1
        end
    end
    res
end

"""
    aggregate_index(ei; direction=:undirected, kind=:edge)

Generate index structure for scatter operation.

# Arguments

- `ei::EdgeIndex`: The reference graph.
- `direction::Symbol`: The direction of an edge to be choose to aggregate. It must be one of `:undirected`, `:inward` and `:outward`.
- `kind::Symbol`: To aggregate feature upon edge or vertex. It must be one of `:edge` and `:vertex`.
"""
function aggregate_index(ei::EdgeIndex, kind::Symbol=:edge, direction::Symbol=:outward)
    # TODO: support CUDA
    if !(direction in [:inward, :outward])
        throw(ArgumentError("direction must be one of :outward or :inward."))
    end

    if kind == :edge
        idx = similar(ei.iadjl, Int, ne(ei))
    elseif kind == :vertex
        idx = similar(ei.iadjl, Vector{Int}, nv(ei))
        for i = 1:length(idx)
            idx[i] = Int[]
        end
    else
        throw(ArgumentError("kind must be one of :edge or :vertex."))
    end
    
    for src_idx = 1:nv(ei)
        for (sink_idx, eidx) in ei.iadjl[src_idx]
            assign_aggr_idx!(Val(kind), Val(direction), idx, src_idx, sink_idx, eidx)
        end
    end
    return idx
end

Zygote.@nograd aggregate_index

assign_aggr_idx!(::Val{:edge}, ::Val{:inward}, idx, src, sink, edge) = (idx[edge] = sink)
assign_aggr_idx!(::Val{:edge}, ::Val{:outward}, idx, src, sink, edge) = (idx[edge] = src)
assign_aggr_idx!(::Val{:vertex}, ::Val{:inward}, idx, src, sink, edge) = push!(idx[src], sink)
assign_aggr_idx!(::Val{:vertex}, ::Val{:outward}, idx, src, sink, edge) = push!(idx[sink], src)

"""
    edge_scatter(aggr, E, ei, direction=:undirected)

Scatter operation for aggregating edge feature into vertex feature.

# Arguments

- `aggr`: aggregating operators, e.g. `+`.
- `E`: Edge features with size of (#feature, #edge).
- `ei::EdgeIndex`: The reference graph.
- `direction::Symbol`: The direction of an edge to be choose to aggregate. It must be one of `:undirected`, `:inward` and `:outward`.
"""
function edge_scatter(aggr, E::AbstractArray, ei::EdgeIndex; direction::Symbol=:undirected)
    if direction == :undirected
        idx1 = aggregate_index(ei, :edge, :outward)
        idx2 = aggregate_index(ei, :edge, :inward)
        # idx may be incomplelely cover all index, this may cause some bug which makes dst shorter than expect
        # currently, scatter idx1 first can avoid this condition
        dst = NNlib.scatter(aggr, E, idx1)
        return NNlib.scatter!(aggr, dst, E, idx2)
    else
        idx = aggregate_index(ei, :edge, direction)
        return NNlib.scatter(aggr, E, idx)
    end
end

"""
    neighbor_scatter(aggr, X, ei, direction=:undirected)

Scatter operation for aggregating neighbor vertex feature together.

# Arguments

- `aggr`: aggregating operators, e.g. `+`.
- `X`: Vertex features with size of (#feature, #vertex).
- `ei::EdgeIndex`: The reference graph.
- `direction::Symbol`: The direction of an edge to be choose to aggregate. It must be one of `:undirected`, `:inward` and `:outward`.
"""
function neighbor_scatter(aggr, X::AbstractArray, ei::EdgeIndex; direction::Symbol=:undirected)
    direction == :undirected && (direction = :outward)
    idx = aggregate_index(ei, :vertex, direction)
    Ys = [neighbor_features(aggr, X, idx[i]) for i = 1:length(idx)]
    return hcat(Ys...)
end

function neighbor_features(aggr, X::AbstractArray{T}, idx) where {T}
    if isempty(idx)
        return fill(NNlib.scatter_empty(aggr, T), size(X, 1))
    else
        return mapreduce(j -> view(X,:,j), aggr, idx)
    end
end
