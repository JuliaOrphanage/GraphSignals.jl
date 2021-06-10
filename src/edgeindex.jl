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

ne(ei::EdgeIndex) = length(unique(map(x -> x[2], vcat(ei.iadjl...))))

neighbors(ei::EdgeIndex, i) = ei.iadjl[i]

get(ei::EdgeIndex, key::NTuple{2}, default=nothing) = _get(ei, key..., default)
get(ei::EdgeIndex, key::CartesianIndex{2}, default=nothing) = _get(ei, key[1], key[2], default)

function _get(ei::EdgeIndex, i, j, default=nothing)
    nbs = neighbors(ei, i)
    # linear search (O(n)) here can be optimized with hash table (O(1))
    for (nbs_j, v) in nbs
        nbs_j == j && return v
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
    generate_cluster_index(E, ei; direction=:undirected)

Generate index structure for scatter operation.
"""
function generate_cluster_index(ei::EdgeIndex; direction::Symbol=:undirected, kind::Symbol=:edge)
    # TODO: support CUDA
    N = if kind == :edge
        ne(ei)
    elseif kind == :vertex
        nv(ei)
    else
        throw(ArgumentError("kind must be one of :edge or :vertex."))
    end
    
    if direction == :undirected
        return undirected_generate_clst_idx(ei)
    elseif direction in [:inward, :outward]
        clst_idx = similar(ei.iadjl, Int, N)
        for src_idx = 1:nv(ei)
            for (sink_idx, eidx) in ei.iadjl[src_idx]
                assign_clst_idx!(Val(kind), Val(direction), clst_idx, src_idx, sink_idx, eidx)
            end
        end
        return clst_idx
    else
        throw(ArgumentError("direction must be one of :undirected, :outward or :inward."))
    end
end

function undirected_generate_clst_idx(ei::EdgeIndex)
    el = [map(x -> (i, x...), ei.iadjl[i]) for i = 1:length(ei.iadjl)]
    el = vcat(el...)
    sort!(el, by=x->x[3])
    unique!(x->x[3], el)
    clst_idx1 = map(x -> x[1], el)
    clst_idx2 = map(x -> x[2], el)
    clst_idx1, clst_idx2
end

assign_clst_idx!(::Val{:edge}, ::Val{:inward}, clst, src, sink, edge) = (clst[edge] = sink)
assign_clst_idx!(::Val{:edge}, ::Val{:outward}, clst, src, sink, edge) = (clst[edge] = src)
assign_clst_idx!(::Val{:vertex}, ::Val{:inward}, clst, src, sink, edge) = (clst[src] = sink)
assign_clst_idx!(::Val{:vertex}, ::Val{:outward}, clst, src, sink, edge) = (clst[sink] = src)

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
        clst_idx1, clst_idx2 = generate_cluster_index(ei, direction=direction)
        # clst_idx may be incomplelely cover all index, this may cause some bug which makes dst shorter than expect
        # currently, scatter clst_idx2 first can avoid this condition
        dst = NNlib.scatter(aggr, E, clst_idx2)
        return NNlib.scatter!(aggr, dst, E, clst_idx1)
    else
        clst_idx = generate_cluster_index(ei, direction=direction)
        return NNlib.scatter(aggr, E, clst_idx)
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
    if direction == :undirected
        clst_idx1, clst_idx2 = generate_cluster_index(ei, direction=direction, kind=:vertex)
        dst = NNlib.scatter(aggr, X, clst_idx2)
        return NNlib.scatter!(aggr, dst, X, clst_idx1)
    else
        clst_idx = generate_cluster_index(ei, direction=direction, kind=:vertex)
        return NNlib.scatter(aggr, X, clst_idx)
    end
end
