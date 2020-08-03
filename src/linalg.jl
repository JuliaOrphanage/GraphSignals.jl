adjacency_matrix(fg::FeaturedGraph, T::DataType=eltype(fg.graph[])) = adjacency_matrix(fg.graph[], T)

function degrees(fg::FeaturedGraph, T::DataType=eltype(fg.graph[]); dir::Symbol=:out)
    degrees(fg.graph[], T; dir=dir)
end

function degree_matrix(fg::FeaturedGraph, T::DataType=eltype(fg.graph[]); dir::Symbol=:out)
    degree_matrix(fg.graph[], T; dir=dir)
end

function inv_sqrt_degree_matrix(fg::FeaturedGraph, T::DataType=eltype(fg.graph[]); dir::Symbol=:out)
    inv_sqrt_degree_matrix(fg.graph[], T; dir=dir)
end

function laplacian_matrix(fg::FeaturedGraph, T::DataType=eltype(fg.graph[]); dir::Symbol=:out)
    laplacian_matrix(fg.graph[], T; dir=dir)
end

function normalized_laplacian(fg::FeaturedGraph, T::DataType=eltype(fg.graph[]); selfloop::Bool=false)
    normalized_laplacian(fg.graph[], T; selfloop=selfloop)
end

function scaled_laplacian(fg::FeaturedGraph, T::DataType=eltype(fg.graph[]))
    scaled_laplacian(fg.graph[], T)
end
