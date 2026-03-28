# Presidents Simple Strategy

This document explains the simple presidents pipeline at a high level, in plain language.

## One-Sentence Summary

We do not classify each sentence alone. We classify each sentence while looking at the sentences around it in the same speech.

## Why This Makes Sense

At first, the task looks like sentence classification:

- take one sentence
- decide whether it is Chirac or Mitterrand

But the data is not really built like that. The sentences come from speeches.

That changes the problem:

- nearby sentences are related
- the speaker usually stays the same for several consecutive sentences
- one ambiguous sentence often becomes easier to classify when you know its context

So the real task is closer to:

- "Given this part of the speech, who is most likely speaking here?"

That is why a speech-level model makes more sense than a sentence-only model.

## Visual Schema

```text
                 SIMPLE PRESIDENTS PIPELINE

   Raw speeches
        |
        v
   Split into sentences
        |
        v
   CamemBERT embeddings
   one vector per sentence
        |
        v
   Group by speech_id
   rebuild each speech as:
   [sentence 1, sentence 2, sentence 3, ...]
        |
        v
   BiLSTM over the whole speech
   each sentence sees left and right context
        |
        v
   Raw probability per sentence
   P(Mitterrand)
        |
        v
   Threshold calibration on labeled CV
   learn best cutoff for macro-F1
        |
        v
   Calibrated probabilities
        |
        v
   Final prediction
   Mitterrand or Chirac
```

## Intuition For Each Block

### 1. Sentence embeddings

We first convert each sentence into a dense vector with CamemBERT.

That vector is a summary of the sentence meaning.

So instead of giving raw text to a small model, we give it a representation that already contains a lot of semantic information.

In simple words:

- CamemBERT helps us understand what the sentence says

### 2. Group by speech

We then regroup the sentence vectors by `speech_id`.

This is important because the neighboring sentences belong together. If we lose the speech structure, we throw away useful information.

In simple words:

- a sentence is easier to classify if we know what comes before and after it

### 3. BiLSTM

The BiLSTM reads the speech from both directions:

- left to right
- right to left

So for each sentence, it can use:

- previous context
- next context

This helps when one sentence alone is ambiguous.

In simple words:

- CamemBERT says what the sentence means
- the BiLSTM says how that sentence fits into the speech

### 4. Raw probability

For every sentence, the model outputs a probability:

- high probability means "looks like Mitterrand"
- low probability means "looks like Chirac"

This is the model's first guess.

### 5. Threshold calibration

The raw probability is not always perfectly calibrated.

A model can be:

- too eager to predict one class
- too conservative

So we do not blindly say:

- above `0.5` => Mitterrand
- below `0.5` => Chirac

Instead, we learn the best cutoff on labeled cross-validation.

This is important because our evaluation metric is macro-F1, not just accuracy.

## Why This Works Well For Our Data

Three reasons matter most.

### 1. The data has local continuity

Inside a speech, the speaker style usually does not change randomly every sentence.

So context helps.

### 2. We already start from strong sentence representations

Because we use CamemBERT embeddings, the model does not need to learn French from scratch.

The BiLSTM only has to learn how labels behave across a sequence of meaningful sentence vectors.

### 3. The classes are imbalanced

There are more examples of one class than the other.

So if we are not careful, the model can overpredict the majority class.

That is why we use:

- class-weighted loss during training
- threshold tuning during calibration

## Why We Use Class Weights

If one class is much more frequent, the model can get good average performance by mostly predicting that class.

But macro-F1 punishes that behavior, because it cares about both classes.

So we use weighted cross-entropy:

- same model
- same data
- but mistakes on the minority class cost more

This pushes the model to pay attention to both speakers.

## Why Threshold Tuning Is Legitimate

This is a normal machine learning step.

We do:

1. grouped cross-validation on labeled speeches
2. collect out-of-fold probabilities
3. try many thresholds
4. keep the one that gives the best macro-F1

This is legitimate because:

- we only use labeled training data
- we never use the hidden test labels

So we are not cheating or adapting to the leaderboard. We are simply calibrating the model correctly for the evaluation metric.

## What `delta` Means

Sometimes the evaluation expects a probability file and then applies a `0.5` cutoff.

If our best threshold is not `0.5`, we can encode that threshold as a score shift:

```text
calibrated_score = sigmoid(logit(raw_score) - delta)
```

You do not need to think of this as something complicated.

It only means:

- the model's raw probabilities were slightly shifted
- the shift makes the final decision rule match the best threshold we learned on labeled data

So:

- threshold tuning is the real idea
- `delta` is just the practical export trick

## Final Pipeline

The simple pipeline is:

1. load sentence embeddings and metadata
2. regroup sentences by `speech_id`
3. train a small speech-level BiLSTM
4. choose the best epoch on one validation split
5. run grouped cross-validation to learn the best threshold
6. retrain on all labeled speeches
7. run on test and export calibrated probabilities

## Why We Chose This Version

We chose this version because it has the best balance between:

- strong performance
- simple code
- easy explanation

It keeps the ideas that clearly helped:

- use speech structure
- use a sequence model
- handle class imbalance
- calibrate the final decision rule

And it removes the extra tricks that were harder to justify for a student project.

## Very Short Oral Version

If you need to explain it in class in a few sentences:

1. Each sentence is converted into a CamemBERT embedding.
2. We regroup the sentences by speech, because isolated sentence classification loses context.
3. A BiLSTM reads the speech and predicts each sentence using both left and right context.
4. We use class-weighted loss because the classes are imbalanced.
5. We tune the final threshold with grouped cross-validation, because `0.5` is not always optimal for macro-F1.
6. We then export calibrated probabilities and final labels for submission.
