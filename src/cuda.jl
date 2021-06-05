promote_graph(graph::AbstractMatrix, nf::CuArray) = cu(graph)

function EdgeIndex(adjl::AbstractVector{<:AnyCuVector{T}}) where {T<:NTuple{2}}
    a = similar(adjl, CuVector{Tuple{Int64, Int64}}, 0)
    a = convert(typeof(a), adjl)
    EdgeIndex{typeof(a)}(a)
end
