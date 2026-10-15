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
LAYOUT := $(shell PYTHONPATH=. $(PYTHON) -c "from blobcount.config import load; from blobcount.registry import load_registry, usable_for_training; c = load(); print(c.path('paths.patches').as_posix(), c.path('paths.labels').as_posix(), c.path('paths.results').as_posix(), ' '.join(sorted(s.id for s in usable_for_training(load_registry()))))")

PATCHES := $(word 1,$(LAYOUT))
LABELS   := $(word 2,$(LAYOUT))
RESULTS  := $(word 3,$(LAYOUT))
ADMITTED := $(wordlist 4,$(words $(LAYOUT)),$(LAYOUT))

# The file names the writers fixed: bin/extract_patches.py writes patches.csv
# into paths.patches and bin/label_patches.py writes labels.csv into paths.labels.
INDEX   := $(PATCHES)/patches.csv
HANDOFF := $(LABELS)/labels.csv

# blobcount.splits numbers the folds over the sorted specimen ids from zero,
# so fold 0 holds out the first admitted specimen and trains on the rest. The
# unprovenanced slides the registry does not clear are never extracted by
# these targets: they are excluded from every reported result (README.md,
# Limitations).
HELD_OUT        := $(word 1,$(ADMITTED))
TRAINED         := $(wordlist 2,$(words $(ADMITTED)),$(ADMITTED))
TRAIN_SPECIMENS := $(foreach s,$(TRAINED),--train-specimens $(s))
CHECKPOINT      := $(RESULTS)/best_model_fold0.pth

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
	$(RUN) bin/label_patches.py --index $(INDEX)

ablate:
	$(RUN) bin/ablate_threshold.py --index $(INDEX)

train:
	$(RUN) bin/train.py --labels $(HANDOFF)

# Scores the fold-0 checkpoint over the specimen that fold held out.
# --train-specimens states the cohort the checkpoint was fitted on, because a
# number computed over a trained specimen is not a measurement.
evaluate:
	$(RUN) bin/evaluate.py --labels $(HANDOFF) --checkpoint $(CHECKPOINT) --specimens $(HELD_OUT) $(TRAIN_SPECIMENS)

# Inference over the patches the labeling rule marked steatotic, with the same
# fold-0 checkpoint; writes $(RESULTS)/predictions.csv.
predict:
	$(RUN) bin/predict.py --dir $(LABELS)/steatosis --checkpoint $(CHECKPOINT)

# One figure per sampled patch of the held-out specimen plus a summary grid,
# from the same fold-0 checkpoint; writes $(RESULTS)/gradcam/.
gradcam:
	$(RUN) bin/gradcam.py --labels $(HANDOFF) --slide $(HELD_OUT) --checkpoint $(CHECKPOINT)

# Removes the results directory and the generated label directory and nothing
# else, through the guarded delete the extraction stage already uses: the
# slides and the manifest are prepare_output's protected paths, so this cannot
# reach them.
clean:
	$(RUN) -c "from blobcount.config import load; from blobcount.extract import prepare_output; cfg = load(); prepare_output(cfg.path('paths.results')); prepare_output(cfg.path('paths.labels'))"
