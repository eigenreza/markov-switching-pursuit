# Data

This page records what each dataset contains, what role it plays, how its recordings become
chases in the common format, and which commands need the raw downloads. Counts come from the
audit files under `results/audit/`, which `python tasks.py audit` regenerates from the raw data.

## Roles

| dataset | role | source | licence |
|---|---|---|---|
| Michaiel, Abe and Niell 2020 | primary predator-prey set: all design choices and the main results | Dryad 10.5061/dryad.8cz8w9gmw (mirror: Zenodo record 3975832) | CC0 |
| Holmgren et al. 2021 | secondary predator-prey set: fitted once with the frozen model as a replication, never used for design choices | Dryad 10.5061/dryad.2z34tmpnc (mirror: Zenodo record 5576696) | CC0 |
| MABe22 mouse triplets | conspecific set: mouse chasing mouse | Caltech Data 10.22002/rdsa8-rde65 | CC BY-NC-SA, as stated in the dataset paper's datasheet |
| CalMS21 task 1 | designated fallback for the conspecific set, not needed and not used | Caltech Data 10.22002/D1.1991 | CC BY 4.0 |

The behaviour deposit of Allen and Hoy (Mendeley Data 10.17632/vvxfszxc88.2) was examined and
is not a data source: its tracked stimulus is a computer-generated ellipse on a wall monitor,
not a live cricket. That study is cited as prior work on mouse prey capture.

## The common format

Every loader produces `Chase` objects defined in `src/aor/contracts.py`: a per-frame table with
the pursuer's position, heading and speed, the target's position and velocity, the range and the
bearing, in centimetres, seconds and radians, plus metadata naming the dataset, session, animal,
frame rate, native frame rate, coordinate frame, end reason and target type. The validator rejects
missing frames, NaN, inconsistent ranges or bearings, coincident positions before the last frame,
an analysis rate above the native rate, and implausible speeds. Pursuer and target positions of one
chase always share one coordinate frame.

Chases are cut from the recordings by one rule (`src/aor/segment.py`): a chase starts when the
pursuer closes on the target faster than 5 cm/s while moving faster than 10 cm/s with the target
within 60 degrees of its heading, from a range of at least 8 cm; it ends at contact (3 cm), after
the range has stopped decreasing for 0.5 s, or when a track is lost; episodes shorter than 0.5 s
are dropped. These thresholds are provisional until the development-set freeze.

## Michaiel, Abe and Niell 2020

Seven mice hunted live crickets in a 45 by 38 cm arena while wearing head-mounted eye cameras
and an inertial sensor. An overhead camera at 30 frames per second was tracked with DeepLabCut,
giving the head centre, the head yaw, and the cricket centre in one pixel frame. The deposited
series were resampled by the authors onto a 60 Hz grid; the loader keeps every second sample
and works at the native 30 Hz. Scale: 27 pixels per centimetre, the authors' calibration, with an
arena-fit alternative carried as a sensitivity check. The pursuer heading is the head yaw, whose
sign convention is calibrated against the direction of travel (median disagreement 12 degrees)
and whose single-frame flips (a sample more than 90 degrees from both neighbours while the
neighbours agree) are replaced by the circular mean of the neighbours; every other sample is
used as measured, because a running median leaves runs of repeated values and so exact zero
increments. The split by
animal is J470c, J462c, J475c and J463b for development and J462b, J462a and J463c held out.

Contents: 105 clips, 105 minutes in total, from 7 animals. The cricket is tracked on about half
of all frames and on nearly every frame within the authors' approach epochs. Seven clips (five
from J470c, two from J475c, 24,324 frames) are excluded because their yaw track flips by more
than 1 radian between consecutive frames on more than 1 percent of frame pairs, a rate of head
rotation a mouse does not produce; the other 98 clips flip on about 0.1 percent. Chases from the
98 clips: 485, median 1.0 s, 231 longer than 1 s, 54 longer than 2 s, of which 64 have a cricket
moving faster than 3 cm/s on average. The development split holds 210 chases with 748 informative
frames and the held-out split 275 chases with 1,022 informative frames. Frames with a moving target are the ones that separate anticipatory from
reactive steering, so the analyses define their informative subset by frame.

## Holmgren et al. 2021

Three mice hunted crickets with olfactory and auditory cues masked; for the fits the recordings are
decimated by seven to 28.6 frames per second so that one delay grid in frames spans the same
seconds as the 30 Hz datasets. Head position and orientation
and the three-dimensional cricket position were tracked at 200 Hz in the centimetre frame of a
laser scan of the arena. The pursuer heading is the horizontal projection of the head's forward
axis, identified by comparison with the direction of travel (median disagreement 10 degrees).
The authors' frame lists for their chase and capture phases give 74 contract-conforming chases;
the shared rule gives 177, of which 120 have a moving cricket.

## MABe22 mouse triplets

Three adult male mice per recording in a 52 by 52 cm arena, filmed at 30 frames per second,
with twelve keypoints per mouse. Chases are derived for every ordered pair of mice by the shared
rule and kept when the target's mean speed exceeds 5 cm/s. Missing markers are stored as (0, 0)
in the source and become NaN. The pixel scale differs between files and is estimated per file
from the extent of the centre-of-back keypoint against the arena size (about 8.2 px/cm for the
training and submission sets, 12.8 px/cm for the test set); the median nose-to-tail-base length
is 5.4 cm in every file after scaling. Mouse identities are not linked across sequences, so
each mouse of each sequence is a separate animal for the splits' summaries and the chase-level
bootstrap strata.

The training and submission files form the development split and the test file the held-out split.
Contents: 6350 one-minute sequences, 106 hours. The frame-level chase label covers 132 bouts,
none longer than 3.3 s, which serve to check the rule (it recovers 82 of them at the provisional
thresholds). The rule yields 13,233 contract-conforming chases with a moving target, median 1.5 s,
4,118 longer than 2 s. Tables derived from this dataset carry their own licence notice.

## Commands and the raw data

- `python tasks.py data` downloads the raw files into `data/raw/` and verifies their hashes.
  Dryad files are fetched from their Zenodo mirrors.
- `python tasks.py audit` needs the raw data and writes `results/audit/<dataset>.json`.
- Model fitting needs the raw data. Tables and figures regenerate from the committed derived
  tables and result files without it.
