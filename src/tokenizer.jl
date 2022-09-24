orthogonal_random_features(nvertex::Int, dims...) =
    orthogonal_random_features(Float32, nvertex, dims...)

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
