promote_graph(graph::AbstractMatrix, nf::AbstractMatrix) = graph
promote_graph(graph::AbstractMatrix, nf::Fill{T,S,Axes}) where {T,S,Axes} = graph
