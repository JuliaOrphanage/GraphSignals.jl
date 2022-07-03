"""
    generate_coordinates(A, with_batch=false)

Returns coordinates for tensor `A`.

# Arguments

- `A::AbstractArray`: The tensor to reference to.
- `with_batch::Bool`: Whether to consider last dimension as batch. If `with_batch=true`,
the last dimension is not consider as a component of coordinates.

# Usage

```jldoctest
julia> using GraphSignals

julia> A = rand(3, 4, 5);

julia> coord = GraphSignals.generate_coordinates(A);

julia> size(coord)
(3, 3, 4, 5)

julia> coord = GraphSignals.generate_coordinates(A, with_batch=true);

julia> size(coord)
(2, 3, 4)
```
"""
function generate_coordinates(A::AbstractArray; with_batch::Bool=false)
    dims = with_batch ? size(A)[1:end-1] : size(A)
    N = length(dims)
    colons = ntuple(i -> Colon(), N)
    coord = similar(A, N, dims...)
    for i in 1:N
        ones = ntuple(x -> 1, i-1)
        coord[i, colons...] .= reshape(1:dims[i], ones..., :)
    end
    return coord
end
