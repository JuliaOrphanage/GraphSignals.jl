N = 6

ug = SimpleGraph(N)
add_edge!(ug, 1, 2); add_edge!(ug, 1, 3); add_edge!(ug, 2, 3)
add_edge!(ug, 3, 4); add_edge!(ug, 2, 5); add_edge!(ug, 3, 6)

dg = SimpleDiGraph(N)
add_edge!(dg, 1, 3); add_edge!(dg, 2, 3); add_edge!(dg, 1, 6)
add_edge!(dg, 2, 5); add_edge!(dg, 3, 4); add_edge!(dg, 3, 5)

el_ug = Vector{Int64}[[2, 3], [1, 3, 5], [1, 2, 4, 6], [3], [2], [3]]
el_dg = Vector{Int64}[[3, 6], [3, 5], [4, 5], [], [], []]

@testset "simplegraph" begin
    fg = FeaturedGraph(ug)
    @test adjacency_list(fg) == el_ug

    fg = FeaturedGraph(dg)
    @test adjacency_list(fg) == el_dg
end
