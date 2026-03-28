# Presidents Simple Strategy

This is the simplest version of the presidents pipeline that still keeps the main performance gains.

## Main Idea

We keep three important ideas:

- group sentences by `speech_id`
- train a small BiLSTM over each speech
- tune the final decision threshold on labeled cross-validation only

We remove the extra tricks that made the code harder to explain:

- span priors
- span smoothing
- position features
- transition features
- leaderboard-based tuning

## Why Speech Grouping Matters

The task is not really “classify one sentence alone”.

Sentences come from speeches, and nearby sentences are strongly related. A BiLSTM can use the left and right context of each sentence inside the speech. That is the main reason it beats a plain sentence-level logistic regression baseline.

## Why BiLSTM Helps

Each sentence already has a transformer embedding, so we do not need to learn text representations from scratch.

The BiLSTM only has to learn how labels evolve inside a speech:

- previous sentences matter
- next sentences matter
- local context helps disambiguate the speaker

This keeps the model small and easier to justify.

## Why Class Weighting Is Needed

The classes are imbalanced. There are many more majority-class sentences than minority-class sentences.

Without class weighting, the model would mostly learn to predict the majority class.

So we use weighted cross-entropy:

- same model
- same labels
- but mistakes on the minority class count more during training

## Why Threshold Tuning Is Legitimate

Raw neural probabilities are often not perfectly calibrated.

So instead of assuming that `0.5` is always the best cutoff, we learn the threshold from labeled data only:

1. run grouped cross-validation on training speeches
2. collect out-of-fold probabilities
3. sweep thresholds
4. keep the threshold that gives the best macro-F1

This is legitimate because the test set is never used to choose the threshold.

## What `delta` Means

The external evaluation often expects one probability file and then applies a `0.5` cutoff.

If our best threshold is not `0.5`, we can convert it into a score shift:

```text
calibrated_score = sigmoid(logit(raw_score) - delta)
```

This is just another way to encode the tuned threshold into the probability file.

So:

- tuning the threshold is the real idea
- `delta` is only the practical way to export that choice

## Final Simple Pipeline

The simple pipeline is:

1. load sentence embeddings and metadata
2. rebuild speeches with `speech_id`
3. train a small speech-level BiLSTM
4. choose the best epoch on a validation split
5. run grouped CV to learn the best threshold
6. retrain on all labeled speeches
7. export calibrated test probabilities and hard labels

This is the version that is easiest to explain while still staying strong.
