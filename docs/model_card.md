# Model card

## What the model is

A hidden Markov model over a small library of guidance laws, fitted to the heading increments
of a pursuer tracked at video frame rate. Each state predicts the next heading change from the
current geometry: pure pursuit (toward the target), lead pursuit (toward the target extrapolated
by a lead time), proportional navigation (in proportion to the rotation of the line of sight),
and, in the post-freeze follow-up, a persistence state (no change). All states share a motor
delay in frames, a residual with first-order autocorrelation and an outlier component. The
output is, for every frame of every chase, a posterior probability over the states, summarised
as the steering probability (the mass of the states whose fitted response to the target is
non-zero), plus the fitted delay, gain, lead time and navigation constant with chase-level
bootstrap intervals.

## What it was built for

Mouse chases from overhead or multi-camera tracking at 28 to 30 frames per second: mouse and
cricket (head yaw as the heading) and mouse and mouse (body axis as the heading). It applies to
any tracked chase written in the common per-frame format described in `docs/data.md`.

## What it was validated on

A synthetic suite with known laws, switches and target behaviours, including a realistic block
matched to the gains, delays and noise of the recordings (`results/recovery/limits.md`). On the
standard block the laws are classified correctly, scripted switches are detected and timed to a
few frames, intervals cover the parameters and the posterior is calibrated. On the realistic
block, per-chase law classification is unreliable and per-chase switching detection is at
chance: claims about real data rest on pooled held-out likelihoods and a dataset-level null.

## What it found

In three datasets the model uses its laws as two behaviours: a steering state (pure pursuit, a
gain near 0.2 per frame, a delay of one to two frames) and a non-steering state (a
navigation constant at zero). Lead pursuit collapses onto pure pursuit. A generic switching
autoregressive model with no law library matches the law model on head yaw and beats it on body
heading.

## Limits and misuse

- The model sees the target's current velocity and nothing richer; "anticipation" means a
  linear extrapolation of it. A pursuer that predicted a manoeuvre from posture would not be
  scored as anticipatory.
- One chase does not identify its law at real-data gains; do not headline per-chase decoded
  laws.
- The heading dynamics of a body axis are richer than a law plus a first-order residual; the
  comparator shows this and the law model should not be read as the best description of
  heading.
- The law library is fixed; adding states changes the comparisons and needs a new recovery
  study.
- The model is not a model of the target's behaviour; the two-player test is a test, not a
  model of evasion.

## Licence and provenance

MIT licence (see `LICENSE`). The recordings belong to their authors and carry their
own licences (CC0 for the two cricket datasets; CC BY-NC-SA for MABe22).
