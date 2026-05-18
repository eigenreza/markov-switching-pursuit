# Methods notes

These notes define the guidance laws, the observation model that the simulator and the
inference share, the switching models, and the optimal-play benchmark, with the sources for
results taken from the literature. The paper's Methods section is written from this page.

## Geometry and notation

At frame t the pursuer is at position p_t with heading psi_t (the direction of its body axis, or
of its head where only the head is tracked) and speed v_t; the target is at x_t with velocity
u_t. The line of sight is the vector from pursuer to target, with angle lambda_t and length r_t
(the range). The bearing is lambda_t minus psi_t, wrapped to (-pi, pi]. Frames are dt = 1/f
apart at f frames per second. Angles are in radians, positive counter-clockwise.

## The law library

Each law maps the geometry at one frame to a commanded heading change c_t in radians per frame.

1. Pure pursuit. The pursuer turns toward the target's current position:
   c_t = g (lambda_t - psi_t), wrapped. The gain g is the fraction of the heading error
   corrected per frame. This is the reactive law: it uses only where the target is now.
2. Lead pursuit. The pursuer turns toward the point the target will reach after a lead time
   tau if it keeps its current velocity, x_t + tau u_t: c_t = g (angle to that point - psi_t),
   wrapped. Pure pursuit is the case tau = 0.
3. Proportional navigation. The pursuer turns in proportion to the rotation of the line of
   sight: c_t = N (lambda_t - lambda_{t-1}), wrapped, with navigation constant N. For this law
   N is the coefficient and no separate gain applies. With large N the line-of-sight rate is
   driven to zero and the bearing is held constant, which is parallel navigation, the classical
   interception law (Shneydor 1998, chapters 3 to 5; Nahin 2007, chapter 3). With N = 1 the
   heading rotates with the line of sight, so an initial bearing is held; when that bearing is
   zero the course coincides with pure pursuit.

Lead pursuit and proportional navigation are anticipatory in the sense used throughout: both
commit to where the target will be under a linear extrapolation of its current velocity. The
model sees that velocity and nothing richer, so "anticipation" here never means prediction of
a target's intentions.

## Shared parameters and the observation model

Every law has a motor delay d (an integer number of frames between the geometry and the heading
response), a steering gain g (for the pursuit laws), and a noise scale sigma on the heading
increment. The observed heading increment is

    psi_{t+1} - psi_t = c_{t-d} + e_t,  e_t ~ Normal(0, sigma^2), wrapped,

where c_{t-d} is the command the active law computed from the geometry d frames earlier. Speed
is not governed by the laws; it enters as an input and is modelled as a first-order
autoregressive process around a mean. The simulator generates data from exactly this model
and, as options the inference does not know about, adds measurement noise on positions and
headings and target behaviours that are not in the inference's repertoire.

## Switching

The active law is a hidden state z_t in {pure pursuit, lead pursuit, proportional navigation}.
Plain switching uses a fixed transition matrix. Input-dependent switching lets the transition
probabilities depend, through a logistic link, on the range, the closing speed, and an indicator
of a target manoeuvre (a sudden change of the target's heading or speed). The simulator also
provides scripted switching (fixed switch frames, switches triggered by manoeuvres, switches
triggered by range) for the recovery study.

## Inference

Each law supplies one regressor per frame: the wrapped bearing error for pure pursuit, the
wrapped error to the lead point for lead pursuit (a function of tau), and the line-of-sight
rotation over the previous frame for proportional navigation. The heading increment at frame
s is modelled as the regressor of the active law at frame s - d, scaled by the gain (pursuit
laws) or by N (proportional navigation), plus Gaussian noise on the wrapped residual with a
common sigma. Rows start at s = d + 1 so that every regressor is defined. The target velocity that enters the
lead-point regressor is the one the pursuer could have perceived at that frame: the backward
difference of the target position over two frames (67 ms at 30 Hz). The simulated pursuer
perceives the target velocity the same way. A centred smoother, which uses frames after the
decision, attenuates the fitted gain at long lead times and is not used for the laws; the
contract's smoothed velocity column serves the manoeuvre indicator and the descriptive analyses.

Two features of tracked headings are built into the residual. First, headings from a body axis or
a head yaw are smooth, so consecutive increments are correlated; the residual of each law carries
a first-order autoregressive term with one shared coefficient, and the likelihood is evaluated on
the innovation. Second, pose tracking produces occasional single-frame glitches; the innovation
is a two-component mixture, a core Gaussian and a wide outlier Gaussian with its own weight and
scale. Both terms are shared by every model, including the persistence null, so that model
comparison concerns the laws alone. Rows start at s = d + 2 so that the lagged residual exists.

A generic comparator with no law library is fitted and scored on the same frames: a switching
model whose three states each predict the heading increment from an intercept and the previous
increment, with the same shared noise model (residual scale, autoregressive coefficient and
outlier component) as the law models. It asks whether the laws add anything beyond a generic
segmentation of the steering signal. The noise model is shared on purpose: tracked headings
repeat exactly on a few percent of frames (pixel quantisation, duplicated tracker output), and a
comparator free to shrink the residual scale of one state would be rewarded for isolating those
frames rather than for explaining steering.

Fitting is by expectation-maximisation. The expectation step runs forward-backward over the
law index for every chase (all chases of a fit in one padded batch). The maximisation step
updates the transition matrix and initial distribution in closed form, the gain and N by
weighted least squares, the lead time by a bounded one-dimensional search, and sigma in closed
form. For input-dependent transitions the transition weights are fitted by a weighted
multinomial logistic regression on the expected transition counts, with the covariates
intercept, scaled range, scaled closing speed, and an indicator raised for a few frames after a
target manoeuvre (a sudden change of the target's heading or speed). The iterations stop when the
log-likelihood improves by less than one part in a million of its value (a looser rule, one part
in ten thousand, stopped the mouse-and-mouse fit while its gain was still rising). Restarts from
perturbed starting points guard against local optima. Single-law models are the same fit with the
posterior pinned to one law; the persistence null has only sigma.

The delay is profiled over 0 to 8 frames (0 to 267 ms at 30 Hz): the model is refitted at each
integer delay on the common support that starts after the largest delay in the grid, and the
delay with the highest marginal likelihood is kept. Model comparison uses the held-out log-likelihood per frame on chases never used in
fitting, with models scored on the same frames. Parameter uncertainty comes from a chase-level
bootstrap, resampling whole chases within animals; intervals are percentile intervals and are
always called bootstrap intervals.

## The recovery study

Every condition of the synthetic suite is split into fitting replicates (even) and held-out
replicates (odd). Held-out chases are classified by the single-law likelihood they score
highest under, less a penalty of half the law's parameter count times the log of the frame
count, because lead pursuit with a zero lead time contains pure pursuit; the switching model's posterior is scored frame by frame against the true law;
the switching gap is the held-out log-likelihood per frame of the switching model minus that
of the best single law, and switching is declared when the gap exceeds the 95th percentile of
the same gap on single-law data. The null for that percentile includes single-law chases with
tracking noise added to positions and headings, because tracking noise is always present in
real data and inflates the gap at low process noise; the percentile from clean single-law
chases alone is reported alongside it. Parameters are recovered on their own conditions with bootstrap
intervals, and the posterior probabilities are binned against the frequency of the true law.
The limits recorded in `results/recovery/limits.md` are the shortest truncation length and the
highest noise at which classification stays at or above the target, the kinds of switching that
are detected at the target rate, the false-detection rate on misspecified targets, the switch
timing error, the interval coverage of each parameter, and the calibration gap.

## The optimal-play benchmark

The simple-motion pursuit game: a pursuer and an evader move in the plane at constant speeds
v_P > v_E and choose their headings freely at every instant; capture occurs when the range falls
to a capture radius l; the pursuer minimises and the evader maximises the time to capture.
The game has a saddle point in which both players run along the line of sight, the evader
directly away from the pursuer and the pursuer directly at the evader, and its value is
(r_0 - l) / (v_P - v_E) (Isaacs 1965, chapter 1, the game of simple pursuit; Başar and Olsder
1999, chapter 8, pursuit-evasion games with simple motion). Under this play the line of sight
keeps a fixed direction, so the pursuer's strategy is the constant-bearing course.

Against a target that does not play its optimal strategy, the constant-bearing course is still
defined by the current geometry: the pursuer heading psi* that nulls the line-of-sight rate
given the target's current velocity satisfies v_P sin(psi* - lambda) = u_perp, where u_perp is
the component of the target's velocity perpendicular to the line of sight. This is the
collision course of parallel navigation (Shneydor 1998, chapter 4), and it reduces to the
line-of-sight direction when the target runs straight away. Where the target's transverse
speed exceeds v_P no such course exists and the line-of-sight direction is used.

The distance from optimal play of one real chase is the time average over the chase of the
absolute wrapped difference between the heading the pursuer actually commanded and psi*
computed from the same geometry, in radians, with a bootstrap interval from resampling frames
within the chase and chases within animals. The optimal capture time for a chase is the time
at which a pursuer running at the observed speed profile on the constant-bearing course against
the recorded target path would first bring the range below the capture radius; it is compared
with the observed capture time where the chase ends in capture.

## Sources

- Isaacs R. Differential Games: A Mathematical Theory with Applications to Warfare and
  Pursuit, Control and Optimization. Wiley, 1965 (reprinted by Dover, 1999).
- Başar T, Olsder GJ. Dynamic Noncooperative Game Theory, second edition. SIAM Classics in
  Applied Mathematics 23, 1999.
- Nahin PJ. Chases and Escapes: The Mathematics of Pursuit and Evasion. Princeton University
  Press, 2007.
- Shneydor NA. Missile Guidance and Pursuit: Kinematics, Dynamics and Control. Horwood, 1998.

## Real-data analysis

The primary dataset is split by animal into a development set, on which every design choice is
made and the models are fitted, and a held-out set scored once. The replication set is fitted once
with the frozen structure. Held-out comparison uses the per-frame log-likelihood on the common
support, with chase-level bootstrap intervals, on all frames and on frames with a moving target.
The null for switching is a parametric bootstrap: the best single law from the development fit is
replayed along the recorded target paths of both splits with the recorded speed profiles and
tracking noise, refitted on the replayed development set, and scored on the replayed held-out set;
the observed held-out gap is compared with that distribution (200 replicates per dataset).
Parameter intervals come from a chase-level bootstrap stratified by animal; every refit starts
from the fitted model with the full settings, so that the interval is built around the point
estimate, and the report records whether it contains it.

The held-out set of the primary dataset was scored once before three preprocessing changes were
made on the development set, and those numbers were discarded: a running median on the head yaw
was replaced by a repair of single-frame flips, because the median left a fifth of heading
increments exactly zero; seven clips whose yaw track flips on more than 1 percent of frame pairs
were excluded; and the generic comparator was given the noise model of the law models, because
with a free scale per state it was rewarded for the repeated samples rather than for the
steering. The held-out numbers reported are from the single pass after these changes.

States are named by what they do, not by the law they were given. A state of a switching fit is
a steering state when its fitted response to the target is at least 0.05 per frame (the gain for
pure and lead pursuit, the navigation constant for proportional navigation) and a non-steering
state otherwise. The reported posterior is the steering probability, the posterior mass of the
steering states; its complement is the non-steering probability. In every dataset the
proportional-navigation state fitted a navigation constant near zero and so served as the
non-steering state. A pre-declared follow-up, run after the freeze and reported after the frozen
analysis, adds an explicit persistence state (zero predicted increment) as a fourth state, with
the comparator given four states as well, and asks whether proportional navigation then takes a
non-zero constant.

Two sensitivity analyses accompany the main results. The replication set, which has no split, is
also scored leave-one-animal-out: every model is refitted on two of the three animals at the
frozen delay and scored on the third, and the per-chase scores are pooled over the folds. The
primary dataset is also run at an arena-fit scale: pixels per centimetre along x and y chosen so
that the extent of the tracked head positions (0.5th to 99.5th percentile over all clips) matches
the stated 45 by 38 cm arena, with the head yaw corrected for the anisotropy, at the frozen delay.

The two-player test treats the target as a player. A logistic regression of manoeuvre onsets on
range, closing speed and the pursuer's filtered steering probability at the previous frame (the
forward pass alone, which uses no observation after that frame; the smoothed posterior would carry
the pursuer's response to the manoeuvre back into its state before it), with chase-level
bootstrap intervals, asks whether manoeuvres depend on the pursuer's state. The null
circularly shifts the steering covariate within each chase. The shift keeps each chase's mean of
the covariate, so the null distribution carries whatever association exists between chases
(chases in which the pursuer rarely steers may also be chases in which the target rarely
manoeuvres); the test statistic is therefore the distance of the observed coefficient from the
centre of the null distribution, the within-chase effect, with a two-sided permutation p-value.
For mouse targets the evader library (pure evasion: away from the pursuer's current position;
lead evasion: away from its extrapolated position; straight flight: no commanded turn) is fitted
to the target's own steering with the same machinery, and the pursuer's steering probability is
cross-correlated over time with the evader's lead-evasion probability against a chase-shuffle
null; the joint occupancy of pursuer state by evader mode is tabulated by fifth of the chase.

Two features of the recordings bear on the interpretation and are stated with the results. The
Michaiel mice wore head-mounted eye cameras and an inertial sensor during the hunts; the
Holmgren mice hunted with olfactory and auditory cues masked. Both conditions could change how
a mouse steers, and the replication across the two datasets is a replication across them.
