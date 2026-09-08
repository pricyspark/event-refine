# Published prompt suites

Commit the exact reviewed prompt suite used by each reported experiment here.
Training data and generated embeddings remain external, but prompt wording is a
small research artifact and must be versioned for reproducibility.

Each file must map every canonical model label to these three strings:

```json
{
  "iris scissors": [
    "iris scissors",
    "This is a photo of iris scissors.",
    "iris scissors with reviewed visible distinguishing features"
  ]
}
```

When a suite began as an API-generated draft, commit its adjacent provenance
JSON too. Report any subsequent manual review or editing in the experiment
documentation. Reproduction should use the committed suite directly rather
than regenerating it.
