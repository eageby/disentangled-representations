.PHONY: %
SHELL := /bin/bash

train: BetaVAE FactorVAE BetaTCVAE

BetaVAE: train/BetaVAE/Shapes3d 
FactorVAE: train/FactorVAE/Shapes3d 
BetaTCVAE: train/BetaTCVAE/Shapes3d 
BetaSVAE: train/BetaSVAE/Shapes3d 

experiments: full_metric

full_metric: full_metric/BetaVAE/Shapes3d
# full_metric: full_metric/FactorVAE/Shapes3d
# full_metric: full_metric/BetaTCVAE/Shapes3d

TESTING_FLAGS=--gin-parameter 'run_training.iterations=100'\
			  # --gin-parameter 'MetricCallback.interval=1000'\
			  --gin-parameter 'mutual_information_gap.batches=10'\
			  --gin-parameter 'factorvae_score.training_votes=10'\
			  --gin-parameter 'factorvae_score.subset=100'\

RANDOM_SEED_LIST = 10 12 33 88 25 42 8 0 99 19
	
train/%:
	disentangled-training --config $@.gin $(TESTING_FLAGS)

repeat/%:
	{ \
    for seed in $(RANDOM_SEED_LIST) ; \
	do make -s FLAGS='--gin-parameter run_training.seed='$${seed} $*;\
	done ;\
    }

full_metric/%:
	disentangled-training --config train/$*.gin --config experiment/full.gin $(TESTING_FLAGS) $(FLAGS)
	
tensorboard:
	tensorboard --logdir $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)logs --reload_multifile=true $(FLAGS)
