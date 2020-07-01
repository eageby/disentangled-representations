SHELL := /bin/bash
VPATH := $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)

DATASETS = DSprites Shapes3d 
METHOD = BetaVAE BetaTCVAE FactorVAE BetaSVAE
MODELS := $(foreach p,$(DATASETS),$(patsubst %,%/$p,$(METHOD)))
METRICS := mig gini_index factorvae_score

RANDOM_SEED_LIST = 10 12 33 #88 25 # 42 8 0 99 19

data: $(patsubst %, serialized/%.tfrecords, $(DATASETS))
train: $(patsubst %, models/%/saved_model.pb, $(MODELS))

experiments: full_metric hyperparameter_search
full_metric: data $(addprefix repeat/full_metric/, $(MODELS))

evaluate: images metrics
images: $(patsubst %, reconstructed/%.png, $(MODELS)) 
metrics: $(foreach p,$(MODELS),$(patsubst %,%/$p.data,$(METRICS)))
	
# TESTING_FLAGS= --gin-parameter 'MetricCallback.interval=1000'\
			  --gin-parameter 'disentangled.metric.mutual_information_gap.batches=1'\
			  --gin-parameter 'factorvae_score.training_votes=10'\
			  --gin-parameter 'factorvae_score.subset=100'\

# Training
# ==============================================================================
train/% models/%/saved_model.pb:
	disentangled train $* $(FLAGS)$(TESTING_FLAGS)

# Evaluate
# ==============================================================================
reconstructed/% reconstructed/%.png: models/%/saved_model.pb
	disentangled evaluate reconstructed  $(FLAGS)

concatenate_images/%: 
	find $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)images/ -type f -name '*.png' -a -name '*$**' ! -wholename '*reconstructed/$*.png' \
	-exec sh -c 'convert -append $$0 $$@ $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)reconstructed/$*.png' {} + \
	-exec sh -c 'echo $$0 $$@ >  $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)reconstructed/$*.order' {} +  

# Metric
# ==============================================================================
metric/gini_index/% gini_index/%.data: models/%/saved_model.pb
	disentangled evaluate $* gini-index $(FLAGS)$(TESTING_FLAGS)

metric/mig/% mig/%.data: models/%/saved_model.pb
	disentangled evaluate $* mig $(FLAGS)$(TESTING_FLAGS)

metric/factorvae_score/% factorvae_score/%.data: models/%/saved_model.pb
	disentangled evaluate $* factorvae-score $(FLAGS)$(TESTING_FLAGS)

# Experiments
# ==============================================================================
repeat/%:
	@{ \
    for seed in $(RANDOM_SEED_LIST) ; \
	do make -s FLAGS='--gin-parameter run_training.seed='$${seed} $*;\
	done ;\
    }

full_metric/%:
	echo $@ | sed -E "s/(full_metric\/)(.+\/)(.+)/\3\.gin/" |  xargs -n 1 -I {} \
		disentangled train $* --config experiment/full_metric.gin --config evaluate/dataset/{} --config evaluate/evaluate.gin $(TESTING_FLAGS)$(FLAGS)

# Other
# ==============================================================================
tensorboard:
	tensorboard --logdir $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)logs --reload_multifile=true $(FLAGS)

clean/logs:
	rm -r -i $(FLAGS) $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)logs
