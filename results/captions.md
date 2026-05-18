# Figure captions

Each figure has an academic caption for the paper and a plain-language explanation for the
demo. Numbers quoted in the paper's captions are taken from the result files at build time,
not typed here.

## Figure 1. Three guidance laws on one geometry

Academic. A pursuer (grey square, arrow for heading) and a moving target (black dot, grey
arrow for its velocity over one lead time; open dot for the lead point). Coloured arrows show
the direction each law commands from this geometry: pure pursuit toward the target's current
position, lead pursuit toward the lead point, and proportional navigation along the
constant-bearing collision course that would null the rotation of the line of sight (dotted).

Plain. Three ways to chase something that moves: run at where it is now (orange), run at where
it will be in a moment (blue), or hold it at a fixed angle in your view so that you meet it
(green). The picture shows the three directions from the same starting situation.

## Figure 2. Simulated chases and the inferred law

Academic. Top: one simulated chase produced by each law (pursuer path in the law's colour,
target path in black, start markers). Bottom: the posterior probability of each law at every
frame from the switching model fitted to held-out chases of the same condition, stacked so that
the three probabilities sum to one.

Plain. Three pretend chases in which we know which rule the chaser followed. Underneath, the
model's moment-by-moment guess about which rule was in use. A good model puts most of the
colour on the rule that generated the chase.

## Figure 3. Recovery on simulated chases

Academic. (A) Classification of held-out single-law chases by the single-law likelihoods,
pooled over noise levels up to 0.1 rad per frame, as row-normalised counts. (B) Decoded minus
true switch time on held-out chases with a scripted switch from lead to pure pursuit; chases
with no decoded switch are counted in the title. (C) Recovered lead time, navigation constant
and delay against their generating values, each with a chase-level bootstrap interval; the
diagonal marks perfect recovery and the title gives interval coverage. (D) Calibration of the
posterior law probabilities: for frames binned by the posterior probability of a law, the
fraction of frames produced by that law.

Plain. How well the method works when we know the truth. A: when chases follow one rule, how
often the method names that rule. B: when the chaser switches rules part way, how close the
method gets to the moment of the switch. C: whether the method recovers the numbers that
describe each rule. D: whether the method's confidence is honest, so that "80 percent" is right
about 80 percent of the time.

## Figure 4. Real chases with the inferred law

Academic. Three held-out chases from the primary dataset chosen for the number of frames with a
moving target. Top: the pursuer's path coloured by the decoded law (Viterbi path of the switching
model fitted on the development animals), the target's path in black, start markers. Middle:
range and target speed. Bottom: the posterior probability of each law. Orange (pure pursuit) and blue (lead pursuit) are the steering states; green, the state given proportional navigation, fits a navigation constant of zero and is the non-steering state.

Plain. Three real chases. The chaser's track changes colour when the model thinks the chaser
changed its rule. The middle row shows how far the target was and how fast it moved; the bottom
row shows how sure the model is about each rule at every moment.

## Figure 5. Model comparison on held-out chases

Academic. Held-out log-likelihood per frame of each model minus that of the best single law,
with chase-level bootstrap intervals (resampling chases within animals), on all frames (black)
and on frames with the target moving faster than 3 cm/s (pink), for each dataset. Models share
the delay, the autoregressive residual and the outlier component; the generic switching
autoregressive comparator shares them and has no law library.

Plain. How well each explanation predicts chases it has never seen. Points to the right of zero
do better than the best single rule. The pink points use only the moments when the target was
moving, which are the moments that tell the rules apart.

## Figure 6. Steering probability around target manoeuvres

Academic. Mean posterior steering probability aligned to the onset of target manoeuvres (a sudden
change of the target's heading or speed) on held-out chases, with a chase-level bootstrap band,
for each dataset; dashed, the curve predicted from chase phase alone. It is flat in the
replication set and noisy in the primary set; in the mice it falls across the manoeuvre, mostly
with the decline of steering over the course of a chase.

Plain. What happens to the chaser's steering right after the target suddenly turns or jumps. In
the cricket chases nothing visible changes. In the mouse chases steering drops, mostly because
chasers steer less as a chase goes on and the target's turns come late in the chase.

## Figure 7. Fitted parameters

Academic. (A) Gain, lead time, navigation constant and delay with chase-level bootstrap
intervals, per dataset. (B) Mean steering probability per animal on frames with a moving
target. (C) Lead time refitted within chase-level tertiles of target speed (circles) and range
(squares), with bootstrap intervals, coloured by dataset as in A.

Plain. The numbers that describe how each chaser steers: how strongly it turns toward its goal,
how far ahead it aims, how long it takes to respond, and whether those numbers change with the
target's speed or distance.

## Figure 8. Distance from optimal play

Academic. (A) Per-chase distance from optimal play (time-averaged absolute angle between the
adopted heading and the constant-bearing course of the simple-motion game) with block-bootstrap
intervals, chases sorted. (B) Observed capture time against the capture time an optimal
pursuer would have achieved along the same target path at the observed speeds, for chases that
ended in contact. (C) The optimal pair of strategies: the pursuer holds the bearing, the evader
runs straight away along the line of sight.

Plain. How close the real chasers come to the textbook best way to catch something. A: the
average angle between where the chaser went and where the best strategy would have sent it. B:
how much longer the real catches took than the best possible.

## Figure 9. The two-player test

Academic. (A) Target manoeuvre rate when the pursuer's decoded state at the previous frame was
steering or non-steering, per dataset; raw rates, before the adjustment for range and closing
speed and the within-chase comparison of the test. (B) Mouse-and-mouse chases: cross-correlation between the
pursuer's steering probability and the evader's lead-evasion probability across lags, with a
chase-shuffle null band. (C) Joint occupancy of pursuer state by evader mode across fifths of the
chase.

Plain. Is the target playing the game too? A: whether targets dodge more when the chaser is
steering at them. B and C: in mouse-and-mouse chases, whether the two animals' states move
together in time.

## Figure 10. Predator-prey versus conspecific chases

Academic. (A) Steering probability per chase (target moving) by dataset. (B) Held-out
evidence for switching relative to the best single law, with bootstrap intervals. (C) The
observed switching gap against its parametric-bootstrap null (the best single law replayed along
the recorded target paths).

Plain. Mice chasing crickets versus mice chasing mice, side by side: how much of the time they
steer at the target, and whether the evidence that they switch between states holds up against
chance.
