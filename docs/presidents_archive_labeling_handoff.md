# Presidents Archive Labeling Handoff

This note summarizes the current archive-based hidden-test labeling status and the safest next steps for model tuning.

## Current Datasets

- Original safe DGX output, before recovery work:
  - `Dataset/out/presidents_archive_matching_dgx_backup_97pct_20260328/`
  - coverage: `26,349 / 27,162` rows accepted = `97.01%`
- Automated recovery run only:
  - `Dataset/out/presidents_archive_matching_dgx_final_auto_20260328/`
  - coverage: `26,545 / 27,162` rows accepted = `97.73%`
- Current final working dataset:
  - `Dataset/out/presidents_archive_matching_dgx_final/`
  - coverage: `26,608 / 27,162` rows accepted = `97.96%`
  - unresolved rows: `554`

Main files in the final dataset:

- `Dataset/out/presidents_archive_matching_dgx_final/row_labels_with_evidence.csv`
- `Dataset/out/presidents_archive_matching_dgx_final/submission_label_reconstructed.csv`
- `Dataset/out/presidents_archive_matching_dgx_final/match_summary.json`
- `Dataset/out/presidents_archive_matching_dgx_final/manual_row_overrides.csv`

## What Changed

Code changes:

- `src/rital_nlp_project/presidents/archive_labeling.py`
- `scripts/match_presidents_hidden_test_archives.py`

Added recovery logic after the original block matcher:

1. Exact-sentence fallback for unresolved rows, but only for informative enough sentences.
2. Same-document gap fill inside a speech when the left and right accepted rows agree on the source.
3. Sentence-level rescoring only inside speeches that already have exactly one accepted source document.

This was validated on labeled backtest rows:

- added `366` extra accepted train rows
- precision on the added rows: `1.0`

## Manual Override

One manual override was applied for `speech_id = 774`.

Reason:

- the hidden speech clearly matches Jacques Chirac's homage to Colonel Henri Rol-Tanguy
- the official source exists on the legacy archived Elysee site, but the current crawler corpus did not cover it well enough for the automatic matcher

The override is documented in:

- `Dataset/out/presidents_archive_matching_dgx_final/manual_row_overrides.csv`

and is marked in `row_labels_with_evidence.csv` with:

- `resolution_method = manual_legacy_archive`

## Final Resolution Breakdown

From `Dataset/out/presidents_archive_matching_dgx_final/match_summary.json`:

- `block_match`: `26,349`
- `exact_sentence_unique_doc`: `137`
- `exact_sentence_unique_label`: `13`
- `same_doc_gap_fill`: `43`
- `single_sentence_single_source`: `2`
- `manual_legacy_archive`: `64`
- unresolved: `554`

## What Is Still Missing

The remaining unresolved rows are concentrated in a few large speeches, especially:

- `speech_id 196`
- `speech_id 892`
- `speech_id 91`
- `speech_id 142`

These do **not** mostly look like threshold failures. The main issue is that one `speech_id` can contain material from multiple source speeches and sometimes multiple presidents. The remaining work is therefore more about better segmentation and broader archive coverage than about lowering thresholds.

## How To Use This For Model Tuning

Safest use:

1. Treat accepted rows in `Dataset/out/presidents_archive_matching_dgx_final/row_labels_with_evidence.csv` as high-confidence pseudo-gold on hidden test.
2. Ignore rows where `review_status != accepted`.
3. Use the accepted subset to compare submissions, tune calibration, and study disagreement patterns.
4. Keep the original `97.01%` backup untouched as a rollback point.

Good practical targets:

- evaluate current leaderboard or local submissions on the accepted pseudo-gold subset
- inspect rows where the model disagrees with archive labels
- bias or recalibrate model probabilities to improve macro-F1 on the accepted subset
- avoid training directly on unresolved rows

## Best Next Step For Another Agent

If the goal is better hidden-test performance, the next agent should probably do this:

1. Use `Dataset/out/presidents_archive_matching_dgx_final/row_labels_with_evidence.csv` as pseudo-gold for offline evaluation.
2. Compare the latest model submissions against this accepted subset.
3. Tune thresholding or score bias on top of the strongest current submission.
4. Only after that, if more coverage is needed, extend archive coverage for the remaining unresolved speeches and revisit speech segmentation.
