using TextClassification
using Documenter

makedocs(;
    modules=[TextClassification],
    authors="Eric S. Tellez",
    repo="https://github.com/sadit/TextClassification.jl/blob/{commit}{path}#L{line}",
    sitename="TextClassification.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sadit.github.io/TextClassification.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sadit/TextClassification.jl",
    devbranch="main",
    branch = "gh-pages",
    versions = ["stable" => "v^", "v#.#.#"]
)
