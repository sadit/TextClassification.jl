var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = TextClassification","category":"page"},{"location":"#TextClassification","page":"Home","title":"TextClassification","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [TextClassification]","category":"page"},{"location":"#TextClassification.MicroTC-Tuple{MicroTC_Config, AbstractVector, Any}","page":"Home","title":"TextClassification.MicroTC","text":"MicroTC(\n    config::MicroTC_Config,\n    train_corpus::AbstractVector,\n    train_y::CategoricalArray;\n    tok=Tokenizer(config.textconfig),\n    verbose=true)\nMicroTC(config::MicroTC_Config, textmodel::TextModel, train_X::AbstractVector{S}, train_y::CategoricalArray; verbose=true) where {S<:SVEC}\n\nCreates a MicroTC model on the given dataset and configuration\n\n\n\n\n\n","category":"method"},{"location":"#TextClassification.predict-Tuple{MicroTC, Any}","page":"Home","title":"TextClassification.predict","text":"predict(tc::MicroTC, text)\npredict(tc::MicroTC, vec::SVEC)\n\nPredicts the label of the given input\n\n\n\n\n\n","category":"method"},{"location":"#TextSearch.vectorize-Tuple{MicroTC, Any}","page":"Home","title":"TextSearch.vectorize","text":"vectorize(tc::MicroTC, text; bow=BOW(), tok=tc.tok, normalize=true)\nvectorize(tc::MicroTC, bow::BOW; normalize=true)\n\nCreates a weighted vector using the model. The input text can be a string or an array of strings; it also can be an already computed bag of words.\n\n\n\n\n\n","category":"method"}]
}
