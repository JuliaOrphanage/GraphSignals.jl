struct EdgeIndex{T}
    adjl::T

    function EdgeIndex(adjl::AbstractVector{<:AbstractVector})
        a = similar(adjl, Vector{NTuple{2}}, 0)
        for x in adjl
            push!(a, x)
        end
        new{typeof(a)}(a)
    end
end

nv(ei::EdgeIndex) = length(ei.adjl)

ne(ei::EdgeIndex) = length(unique(map(x -> x[2], vcat(ei.adjl...))))

neighbors(ei::EdgeIndex, i) = ei.adjl[i]

function get(ei::EdgeIndex, key::NTuple{2}, default=nothing)
    i, j = key
    nbs = neighbors(ei, i)
    # linear search (O(n)) here can be optimized with hash table (O(1))
    for (nbs_j, v) in nbs
        nbs_j == j && return v
    end
    return default
end