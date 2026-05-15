# Metadata arguments for choice structure

choicekit's MVP represents Choice Set, Alternative, and Individual identifiers as explicit metadata arguments rather than feature columns in `X` or a public dataset object. This keeps model features compatible with scikit-learn-style preprocessing while preventing identifiers from being accidentally estimated as numeric variables, and it leaves room for a private normalized data representation without committing to a public `ChoiceDataset` API before the core estimator semantics are proven.
