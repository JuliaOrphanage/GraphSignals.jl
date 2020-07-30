using GraphSignals
using Documenter

makedocs(;
    modules=[GraphSignals],
    authors="Yueh-Hua Tu",
    repo="https://github.com/yuehhua/GraphSignals.jl/blob/{commit}{path}#L{line}",
    sitename="GraphSignals.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://yuehhua.github.io/GraphSignals.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/yuehhua/GraphSignals.jl",
)
