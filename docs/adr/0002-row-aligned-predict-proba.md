# Row-aligned predicted probabilities

choicekit's MVP returns one probability per Alternative Row from `predict_proba`, aligned exactly to the input row order and normalized within each Choice Set. This deliberately deviates from scikit-learn's usual classifier shape of `(n_samples, n_classes)` because long-format choice data can have unbalanced Choice Sets, Alternatives are not ordinary classifier classes, and the natural prediction target is the probability that each Alternative Row is the Chosen Alternative.
