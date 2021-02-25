
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sadit.github.io/TextClassification.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sadit.github.io/TextClassification.jl/dev)
[![Build Status](https://github.com/sadit/TextClassification.jl/workflows/CI/badge.svg)](https://github.com/sadit/TextClassification.jl/actions)
[![Coverage](https://codecov.io/gh/sadit/TextClassification.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sadit/TextClassification.jl)                        

# TextClassification.jl

This package provides methods to create fast and simple text classifiers, based on the same idea behind [MicroTC](https://github.com/INGEOTEC/microtc).
The main idea is to perform a model selection among a large space of configurations, including preprocessing steps, weighting schemes, tokenizers (combinations), and classifiers. Moreover, `TextClassification.jl` also includes support for different classifiers and fine-tune them in the search stage; additional support for weighthing shcmes, and a better support for distributed computing thanks to Julia. As the original implementation, this package is designed to be both domain and language independent.


