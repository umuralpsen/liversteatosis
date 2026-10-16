# Reproduction harness for the liver steatosis pipeline.
#
# This file names no directory and no specimen. Every path below is read from
# configs/default.yaml and the training cohort from data/manifest.yaml, both
# through the project's own code, so the Makefile only invokes the bin/ entry
# points and the configuration decides where they read and write.
# `make all` runs the five stages in the order their handoffs connect them and
# is the single command that reproduces every number reported in README.md.

PYTHON ?= python
RUN = PYTHONPATH=. $(PYTHON)

# The facts every stage below needs, from one read of one configuration: the
# patch and label directories (paths.patches and paths.labels, whose %VAR% forms
# blobcount.config expands), the results directory, and the specimens the
# manifest clears for training (blobcount.registry.usable_for_training: a
# source case id and a licence). as_posix() so the paths survive the POSIX
# shell a make recipe runs through, whatever platform the config names them on.
#
# **The three directories are printed with every space in them written as '@', and every
# recipe quotes what it passes on.** `$(word N,...)` below splits on whitespace and on
# nothing else, so a path holding a space is two records and `PATCHES` becomes the first
# half of a directory: a `paths.patches` of `.../my slides/patches` gives `.../my`, and
# every recipe reads a path that does not exist. Those three values are hand-edited strings
# in a git-tracked file and they name the 19 GB of generated data outside this checkout, so
# the failure is neither hypothetical nor cheap.
#
# **The records stay space-separated — the substitution is inside the record, not
# instead of one.** That is the whole of the fix and it is why a second separator
# character is not wanted: make does not split on anything but whitespace, so a `|`
# between two records would not separate them at all, it would glue them into one word.
# A record that carries no whitespace is what `$(word N,...)` needs, and a placeholder
# is how a path that legitimately carries one is made to comply.
#
# **The quoting is the second half and was added on 2026-10-16, because the substitution
# alone only got the value as far as make.** make expands a recipe and hands the line to
# `sh`, and `sh` splits it again: `--index C:/data/my patches/patches/patches.csv` arrived
# as `--index`, `C:/data/my` and `patches/patches/patches.csv`. A variable that reads
# correctly in `make print` and is split by the shell is the same defect one layer down,
# and the test that could see it has to run a recipe rather than print a variable —
# `tests/test_makefile.py::test_a_configured_path_containing_a_space_reaches_the_shell_as_
# one_argument` does, for all six recipes that hand a path to a program.
#
# The ids need none of this: a specimen id carries no space, so `ADMITTED` below is cut
# out of the same list untouched. **'@' is the one character this transport cannot
# carry**, and a configured path containing it would be mangled; it is a legal path
# character in principle and not one a directory this project names ever uses.
LAYOUT := $(shell PYTHONPATH=. $(PYTHON) -c "from blobcount.config import load; from blobcount.registry import load_registry, usable_for_training; c = load(); print(' '.join(p.replace(' ', '@') for p in (c.path('paths.patches').as_posix(), c.path('paths.labels').as_posix(), c.path('paths.results').as_posix())) + ' ' + ' '.join(sorted(s.id for s in usable_for_training(load_registry()))))")

# One `$(subst)` per field, and it is the only place a space is put back. Recursively
# assigned (`=`, not `:=`) on purpose, and it is not a style preference: with `:=` each
# value is frozen when the Makefile is read, so a caller that includes this file and
# supplies its own LAYOUT gets the real configuration's paths back. That is what
# `tests/test_makefile.py` does to exercise the real derivations over a layout carrying
# a space in every field, which is the claim this transport has to hold — and it is why
# every variable below, `INDEX` and `HANDOFF` included, is `=` and not `:=`.
PATCHES = $(subst @, ,$(word 1,$(LAYOUT)))
LABELS   = $(subst @, ,$(word 2,$(LAYOUT)))
RESULTS  = $(subst @, ,$(word 3,$(LAYOUT)))
ADMITTED = $(wordlist 4,$(words $(LAYOUT)),$(LAYOUT))

# The file names the writers fixed: bin/extract_patches.py writes patches.csv
# into paths.patches and bin/label_patches.py writes labels.csv into paths.labels.
#
# **Recursively assigned (`=`), like the three above them, and for the same reason.**
# These two were `:=` until 2026-10-16 and were frozen with the real configuration's paths
# the moment this file was read, so an `override LAYOUT` could not reach them and the live
# probe read the real `C:/Users/.../patches` where it meant to read the synthetic one. That
# is the failure this file's transport exists to prevent, sitting one line below the fix
# for it.
INDEX   = $(PATCHES)/patches.csv
HANDOFF = $(LABELS)/labels.csv

# blobcount.splits numbers the folds over the sorted specimen ids from zero,
# so fold 0 holds out the first admitted specimen and trains on the rest. The
# unprovenanced slides the registry does not clear are never extracted by
# these targets: they are excluded from every reported result (README.md,
# Limitations).
#
# **The ids are cut out bare, with no `$(subst)` and no quoting in the recipes, and
# both are deliberate.** An id carries no space, so there is nothing for a `subst` to put
# back — the `@` transport above is a workaround for a character ids do not have — and
# `--train-specimens` has to expand to several separate words for the programs to parse it.
# The `$(subst |, ,...)` these two lines carried until 2026-10-16 was therefore a no-op
# on every value that can reach them, and a no-op that reads as a second transport is worse
# than none: it says a `|` can appear in a specimen id, which no id this project names can.
HELD_OUT        = $(word 1,$(ADMITTED))
TRAINED         = $(wordlist 2,$(words $(ADMITTED)),$(ADMITTED))
TRAIN_SPECIMENS = $(foreach s,$(TRAINED),--train-specimens $(s))
CHECKPOINT      = $(RESULTS)/best_model_fold0.pth

.PHONY: all install test extract label ablate train evaluate predict gradcam clean

all: extract label ablate train evaluate

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(RUN) -m pytest

# --force discards a previous run's patch tree, because `make all` is the
# reproduction command and has to run on a machine that already holds one. The
# deletion goes through blobcount.extract.prepare_output, which refuses the
# slides, the manifest, and anything this project does not own before it clears.
extract:
	$(RUN) bin/extract_patches.py $(foreach s,$(ADMITTED),--slide $(s)) --force

label:
	$(RUN) bin/label_patches.py --index "$(INDEX)"

ablate:
	$(RUN) bin/ablate_threshold.py --index "$(INDEX)"

train:
	$(RUN) bin/train.py --labels "$(HANDOFF)"

# Scores the fold-0 checkpoint over the specimen that fold held out.
# --train-specimens states the cohort the checkpoint was fitted on, because a
# number computed over a trained specimen is not a measurement, and
# --selection-cohort states the specimen whose validation F1 chose the epoch
# this checkpoint holds, because the program loads a state dict and cannot know.
# For fold 0 those are the same specimen, and the program is told so rather than
# left to work it out from the file name.
#
# **The two ids are bare and the two paths are quoted**, which is the same division the
# definitions above draw: an id carries no space and `--train-specimens` has to expand to
# several words, while a path can carry one and the shell would split it into two
# arguments the program then refuses as a directory that does not exist.
evaluate:
	$(RUN) bin/evaluate.py --labels "$(HANDOFF)" --checkpoint "$(CHECKPOINT)" --specimens $(HELD_OUT) $(TRAIN_SPECIMENS) --selection-cohort $(HELD_OUT)

# Inference over the patches the labeling rule marked steatotic, with the same
# fold-0 checkpoint; writes $(RESULTS)/predictions.csv.
#
# **--specimen $(HELD_OUT) and --train-specimens are the two halves of the same
# correction.** The steatosis directory holds every admitted specimen's steatotic
# patches, and the fold-0 checkpoint was fitted on two of the three, so a run
# without the filter wrote 3079 of 3502 rows over specimens the model had already
# seen - resubstitution, into a tracked artifact, with no checkpoint named, no
# specimen column and no flag. The filter points the run at the specimen fold 0
# held out; --train-specimens marks every row as trained-on or not whatever the
# filter did, so the CSV stays readable by anyone who joins it to labels.csv.
#
# **The `--dir` value is assembled and quoted here** rather than through a variable,
# because it is the only path in this file that is a directory joined to a subdirectory
# and it is the one an unquoted recipe split in two first.
predict:
	$(RUN) bin/predict.py --dir "$(LABELS)/steatosis" --checkpoint "$(CHECKPOINT)" --labels "$(HANDOFF)" --specimen $(HELD_OUT) $(TRAIN_SPECIMENS)

# One figure per sampled patch of the held-out specimen plus a summary grid,
# from the same fold-0 checkpoint; writes $(RESULTS)/gradcam/.
gradcam:
	$(RUN) bin/gradcam.py --labels "$(HANDOFF)" --slide $(HELD_OUT) --checkpoint "$(CHECKPOINT)"

# Removes the results directory and the generated label directory and nothing
# else, through the guarded delete the extraction stage already uses: the
# slides and the manifest are prepare_output's protected paths, so this cannot
# reach them.
clean:
	$(RUN) -c "from blobcount.config import load; from blobcount.extract import prepare_output; cfg = load(); prepare_output(cfg.path('paths.results')); prepare_output(cfg.path('paths.labels'))"
