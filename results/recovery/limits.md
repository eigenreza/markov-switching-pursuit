# Recovery limits

Noise is the standard deviation of the heading-increment noise in radians per frame; length is in frames at 30 per second.

## Law classification of held-out single-law chases

| noise | full length | 30 frames | 60 frames | 120 frames | 180 frames |
|---|---|---|---|---|---|
| 0.02 | 0.950 | 0.817 | 0.982 | 0.944 | 1.000 |
| 0.05 | 1.000 | 0.967 | 1.000 | 1.000 | 1.000 |
| 0.1 | 1.000 | 0.966 | 1.000 | 1.000 | 1.000 |
| 0.2 | 1.000 | 0.932 | 0.979 | 1.000 | 1.000 |

Accuracy target 0.9. Shortest length reaching the target, by noise: noise 0.02: 60 frames, noise 0.05: 30 frames, noise 0.1: 30 frames, noise 0.2: 30 frames.
Highest noise reaching the target, by length: 30 frames: 0.2, 60 frames: 0.2, 120 frames: 0.2, 180 frames: 0.2.

## Posterior law accuracy of one model fitted across the three laws

- noise 0.02, lead pursuit: 0.909 of held-out frames assigned to the true law
- noise 0.02, proportional navigation: 1.000 of held-out frames assigned to the true law
- noise 0.02, pure pursuit: 1.000 of held-out frames assigned to the true law
- noise 0.05, lead pursuit: 1.000 of held-out frames assigned to the true law
- noise 0.05, proportional navigation: 1.000 of held-out frames assigned to the true law
- noise 0.05, pure pursuit: 1.000 of held-out frames assigned to the true law
- noise 0.1, lead pursuit: 1.000 of held-out frames assigned to the true law
- noise 0.1, proportional navigation: 1.000 of held-out frames assigned to the true law
- noise 0.1, pure pursuit: 1.000 of held-out frames assigned to the true law
- noise 0.2, lead pursuit: 1.000 of held-out frames assigned to the true law
- noise 0.2, proportional navigation: 0.988 of held-out frames assigned to the true law
- noise 0.2, pure pursuit: 1.000 of held-out frames assigned to the true law

## Switching detection

Detection threshold: the 0.95 quantile of the held-out switching gap on the null, which is single, misspec_measurement.

- switch fixed at noise 0.05: detected in 0.95 of held-out chases
- switch fixed at noise 0.1: detected in 1.00 of held-out chases
- switch manoeuvre at noise 0.05: detected in 0.00 of held-out chases
- switch manoeuvre at noise 0.1: detected in 0.00 of held-out chases
- switch manoeuvre turn at noise 0.05: detected in 1.00 of held-out chases
- switch manoeuvre turn at noise 0.1: detected in 1.00 of held-out chases
- switch markov at noise 0.05: detected in 1.00 of held-out chases
- switch markov at noise 0.1: detected in 1.00 of held-out chases
- switch range at noise 0.05: detected in 1.00 of held-out chases
- switch range at noise 0.1: detected in 0.95 of held-out chases

False detection on misspecified targets outside the null (evader and jumper): noise 0.05: 0.00, noise 0.1: 0.02.
Detection target 0.8; false-detection bound 0.1. Kinds meeting the target: switch_fixed@0.05, switch_fixed@0.1, switch_manoeuvre_turn@0.05, switch_manoeuvre_turn@0.1, switch_markov@0.05, switch_markov@0.1, switch_range@0.05, switch_range@0.1.

## Switch timing

Scripted lead-to-pure switch: median absolute error 2 frames, 0.86 of decoded switches within 5 frames, 3 of 40 held-out chases with no decoded switch.

## Parameter recovery

- N: bootstrap-interval coverage 0.92 over 13 conditions, mean absolute error 0.068
- delay: bootstrap-interval coverage 1.00 over 15 conditions, mean absolute error 0.000
- gain: bootstrap-interval coverage 0.83 over 18 conditions, mean absolute error 0.004
- tau: bootstrap-interval coverage 1.00 over 13 conditions, mean absolute error 0.001
- delay: exact in 1.00 of 15 conditions, within one frame in 1.00

## The realistic regime

Conditions with the per-frame gain, delay, noise correlation, tracking noise and glitches seen in the real recordings (gain 0.05, delay 3 frames).

- realistic single, full length: 0.70 of 60 held-out chases classified correctly by their own likelihoods
- realistic single, 30 frames: 0.49 of 59 held-out chases classified correctly by their own likelihoods
- realistic single, 60 frames: 0.50 of 54 held-out chases classified correctly by their own likelihoods
- realistic single, 120 frames: 0.53 of 43 held-out chases classified correctly by their own likelihoods
- realistic single, 180 frames: 0.72 of 25 held-out chases classified correctly by their own likelihoods
- realistic still, full length: 0.45 of 60 held-out chases classified correctly by their own likelihoods
- realistic still, 30 frames: 0.42 of 60 held-out chases classified correctly by their own likelihoods
- realistic still, 60 frames: 0.45 of 55 held-out chases classified correctly by their own likelihoods
- realistic still, 120 frames: 0.60 of 30 held-out chases classified correctly by their own likelihoods
- realistic single, one model pooled over the three laws: 0.47 of held-out frames and 0.53 of chases assigned to the true law; recovered gain 0.051, lead time 0.59 s, N 0.64, delay 5 frames; posterior occupancy 0.43, 0.39, 0.17 against a true mix of 0.33, 0.33, 0.33
- realistic still, one model pooled over the three laws: 0.32 of held-out frames and 0.33 of chases assigned to the true law; recovered gain 0.082, lead time 0.28 s, N 0.14, delay 3 frames; posterior occupancy 0.33, 0.27, 0.39 against a true mix of 0.33, 0.33, 0.33
- realistic markov: switching detected in 0.00 of 20 held-out chases against the realistic single-law null (threshold 0.0524 per frame)
- realistic switch: switching detected in 0.10 of 20 held-out chases against the realistic single-law null (threshold 0.0524 per frame)

## Calibration

Weighted absolute gap between posterior probability and empirical frequency: 0.013; largest gap in any bin 0.328.
