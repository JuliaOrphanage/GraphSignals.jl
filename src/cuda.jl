promote_graph(graph::AbstractMatrix, nf::CuArray) = cu(graph)

function EdgeIndex(adjl::AbstractVector{T}) where {T<:AnyCuVector}
    a = similar(adjl, CuVector{Tuple{Int64, Int64}}, 0)
    a = convert(typeof(a), adjl)
    EdgeIndex{typeof(a)}(a)
end
