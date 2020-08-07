adjacency_matrix(fg::FeaturedGraph, T::DataType=eltype(graph(fg))) = adjacency_matrix(graph(fg), T)

function degrees(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); dir::Symbol=:out)
    degrees(graph(fg), T; dir=dir)
end

function degree_matrix(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); dir::Symbol=:out)
    degree_matrix(graph(fg), T; dir=dir)
end

function inv_sqrt_degree_matrix(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); dir::Symbol=:out)
    inv_sqrt_degree_matrix(graph(fg), T; dir=dir)
end

function laplacian_matrix(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); dir::Symbol=:out)
    laplacian_matrix(graph(fg), T; dir=dir)
end

function normalized_laplacian(fg::FeaturedGraph, T::DataType=eltype(graph(fg)); selfloop::Bool=false)
    normalized_laplacian(graph(fg), T; selfloop=selfloop)
end

function scaled_laplacian(fg::FeaturedGraph, T::DataType=eltype(graph(fg)))
    scaled_laplacian(graph(fg), T)
end
