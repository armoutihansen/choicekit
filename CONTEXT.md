# choicekit

choicekit is a Python package context for modeling choices from long-format random utility data. Its language distinguishes the row shape of the data from the statistical decision instances being modeled.

## Language

**Choice Set**:
One observed decision instance containing two or more alternatives, exactly one of which is chosen.
_Avoid_: Case, choice situation, group, sample

**Alternative Row**:
One long-format row representing a candidate alternative within a Choice Set.
_Avoid_: Sample, observation

**Alternative**:
The real-world option identified within a Choice Set, which may appear across many Choice Sets.
_Avoid_: Class, label

**Chosen Alternative**:
The Alternative Row marked as selected within a Choice Set.
_Avoid_: Positive class

**Individual**:
An optional entity associated with one or more Choice Sets.
_Avoid_: Person, customer, panel unit

**Long-Format Choice Data**:
Tabular data where rows are Alternative Rows and Choice Sets are represented by metadata.
_Avoid_: Wide choice matrix

## Relationships

- A **Choice Set** contains two or more **Alternative Rows**.
- A **Choice Set** has exactly one **Chosen Alternative**.
- An **Alternative Row** refers to exactly one **Alternative**.
- An **Individual** may be associated with zero or more **Choice Sets**.
- **Long-Format Choice Data** represents **Choice Sets** as collections of **Alternative Rows**.

## Example Dialogue

> **Dev:** "Should `predict_proba` return probabilities per sample?"
> **Domain expert:** "In this context, the rows are **Alternative Rows**. Return one probability per **Alternative Row**, normalized within each **Choice Set**."

## Flagged Ambiguities

- "sample" is ambiguous between a scikit-learn row and a statistical decision instance; resolved: use **Alternative Row** for rows and **Choice Set** for decision instances.
- "class" is ambiguous with classifier labels; resolved: use **Alternative** for choice options.
