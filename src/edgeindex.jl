struct EdgeIndex{T<:AbstractVector{<:AbstractVector}}
    adjl::T
end

function EdgeIndex(adjl::AbstractVector{T}) where {T<:Vector}
    a = convert(Vector{Vector{Tuple{Int64, Int64}}}, adjl)
    EdgeIndex{typeof(a)}(a)
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
