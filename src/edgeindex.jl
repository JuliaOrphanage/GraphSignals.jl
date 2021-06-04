"""
A indexing structure for accessing neighbors of a vertex. 
"""
struct EdgeIndex{T<:AbstractVector{<:AbstractVector}}
    adjl::T
end

function EdgeIndex(adjl::AbstractVector{T}) where {T<:Vector}
    a = convert(Vector{Vector{Tuple{Int64, Int64}}}, adjl)
    EdgeIndex{typeof(a)}(a)
end

function EdgeIndex(fg::FeaturedGraph)
    iadjl = order_edges(adjacency_list(fg), directed=fg.directed)
    EdgeIndex(iadjl)
end

nv(ei::EdgeIndex) = length(ei.adjl)

ne(ei::EdgeIndex) = length(unique(map(x -> x[2], vcat(ei.adjl...))))

neighbors(ei::EdgeIndex, i) = ei.adjl[i]

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
function generate_cluster_index(E::AbstractArray, ei::EdgeIndex; direction::Symbol=:undirected)
    if direction == :undirected
        return undirected_generate_clst_idx(E, ei, ne(ei))
    elseif direction == :inward
        return inward_generate_clst_idx(E, ei, ne(ei))
    elseif direction == :outward
        return outward_generate_clst_idx(E, ei, ne(ei))
    else
        throw(ArgumentError("direction must be one of :undirected, :outward or :inward."))
    end
end

function undirected_generate_clst_idx(E::AbstractArray, ei::EdgeIndex, num_E::Int=ne(ei))
    clst_idx1 = similar(E, Int, num_E)
    clst_idx2 = similar(E, Int, num_E)
    viewed = Set{Int}()
    for i = 1:nv(ei)
        for (vidx, eidx) in ei.adjl[i]
            if eidx in viewed
                clst_idx2[eidx] = vidx
            else
                clst_idx1[eidx] = vidx
                push!(viewed, eidx)
            end
        end
    end
    clst_idx1, clst_idx2
end

function inward_generate_clst_idx(E::AbstractArray, ei::EdgeIndex, num_E::Int=ne(ei))
    clst_idx = similar(E, Int, num_E)
    # inward vertex index and edge index
    for i = 1:nv(ei)
        for (vidx, eidx) in ei.adjl[i]
            clst_idx[eidx] = vidx
        end
    end
    clst_idx
end

function outward_generate_clst_idx(E::AbstractArray, ei::EdgeIndex, num_E::Int=ne(ei))
    clst_idx = similar(E, Int, num_E)
    # outward vertex index and edge index
    for vidx = 1:nv(ei)
        for (_, eidx) in ei.adjl[vidx]
            clst_idx[eidx] = vidx
        end
    end
    clst_idx
end

"""
    edge_scatter(aggr, E, ei, direction=:undirected)

Scatter operation for aggregating edge feature into vertex feature.
"""
function edge_scatter(aggr, E::AbstractArray, ei::EdgeIndex; direction::Symbol=:undirected)
    if direction == :undirected
        clst_idx1, clst_idx2 = generate_cluster_index(E, ei, direction=direction)
        dst = NNlib.scatter(aggr, E, clst_idx1)
        return NNlib.scatter!(aggr, dst, E, clst_idx2)
    else
        clst_idx = generate_cluster_index(E, ei, direction=direction)
        return NNlib.scatter(aggr, E, clst_idx)
    end
end
