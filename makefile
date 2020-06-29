SHELL := /bin/bash
VPATH := $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)

DATASETS = DSprites Shapes3d 
METHOD = BetaVAE BetaTCVAE FactorVAE BetaSVAE
MODELS := $(foreach p,$(DATASETS),$(patsubst %,%/$p,$(METHOD)))
METRICS := mig gini_index factorvae_score

RANDOM_SEED_LIST = 10 12 33 88 25 # 42 8 0 99 19

data: $(patsubst %, serialized/%.tfrecords, $(DATASETS))
train: $(patsubst %, models/%/saved_model.pb, $(MODELS))
	echo $^

experiments: full_metric hyperparameter_search
full_metric: data $(addprefix repeat/full_metric/, $(MODELS))

evaluate: images metrics
images: $(patsubst %, images/%.png, $(MODELS)) 
metrics: $(foreach p,$(MODELS),$(patsubst %,%/$p.data,$(METRICS)))
	
# TESTING_FLAGS= --gin-parameter 'MetricCallback.interval=1000'\
			  --gin-parameter 'disentangled.metric.mutual_information_gap.batches=1'\
			  --gin-parameter 'factorvae_score.training_votes=10'\
			  --gin-parameter 'factorvae_score.subset=100'\

# Training
# ==============================================================================
train/% models/%/saved_model.pb:
	disentangled-training --config $*.gin $(FLAGS) $(TESTING_FLAGS)

# Evaluate
# ==============================================================================
images/% images/%.png:  
	disentangled-template --config evaluate/visual/images.gin --config evaluate/model/$*.gin $(FLAGS)

concatenate_images/%: 
	find $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)images/ -type f -name '*.png' -a -name '*$**' ! -wholename '*images/$*.png' \
	-exec sh -c 'convert -append $$0 $$@ $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)images/$*.png' {} + \
	-exec sh -c 'echo $$0 $$@ >  $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)images/$*.order' {} +  

# Metric
# ==============================================================================
metric/gini_index/% gini_index/%.data: models/%/saved_model.pb
	disentangled-template --config evaluate/metric/gini.gin --config evaluate/model/$*.gin $(FLAGS) $(TESTING_FLAGS)

mig/% mig/%.data: models/%/saved_model.pb
	disentangled-template --config evaluate/metric/mig.gin --config evaluate/model/$*.gin $(FLAGS) $(TESTING_FLAGS)

factorvae_score/% factorvae_score/%.data: models/%/saved_model.pb
	disentangled-template --config evaluate/metric/factorvae_score.gin --config evaluate/model/$*.gin $(FLAGS) $(TESTING_FLAGS)

# Experiments
# ==============================================================================
repeat/%:
	@{ \
    for seed in $(RANDOM_SEED_LIST) ; \
	do make -s FLAGS='--gin-parameter run_training.seed='$${seed} $*;\
	done ;\
    }

full_metric/%:
	echo $@ | sed -E "s/(full_metric\/)(.+\/)(.+)/\1\3\.gin/" | xargs -n 1 -I {} \
		disentangled-training --config train/$*.gin --config experiment/{} $(TESTING_FLAGS) $(FLAGS)	

# Data
# ==============================================================================
serialize/%.tfrecords:
	disentangled-prepare-data --config serialize/$*.gin $(FLAGS)
	
# Other
# ==============================================================================
tensorboard:
	tensorboard --logdir $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)logs --reload_multifile=true $(FLAGS)

clean/logs:
	rm -r -i $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)logs


