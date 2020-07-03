SHELL := /bin/bash
VPATH := $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)

DATASETS = DSprites Shapes3d 
METHOD = BetaVAE BetaTCVAE FactorVAE BetaSVAE
MODELS := $(foreach p,$(DATASETS),$(patsubst %,%/$p,$(METHOD)))
METRICS := mig gini_index factorvae_score


evaluate: images metrics	

# Training
# ==============================================================================
train: $(patsubst %, models/%/saved_model.pb, $(MODELS))
models/%/saved_model.pb:
	disentangled train $* $(FLAGS)

# Images
# ==============================================================================
images: examples econstructed concatenated

reconstructed: $(patsubst %, images/reconstructed/%.png, $(MODELS))
images/reconstructed/%.png: models/%/saved_model.pb
	disentangled evaluate reconstructed  $(FLAGS)

concatenate_images: $(patsubst %, concatenate_images/%, $(DATASETS))
concatenate_images/%: reconstructed
	find $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)images/ -type f -name '*.png' -a -name '*$**' ! -wholename '*reconstructed/$*.png' \
	-exec sh -c 'convert -append $$0 $$@ $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)reconstructed/$*.png' {} + \
	-exec sh -c 'echo $$0 $$@ >  $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)reconstructed/$*.order' {} +  

examples: $(patsubst %, images/dataset/%.png, $(DATASETS))
images/dataset/%.png:
	disentangled dataset $* examples  $(FLAGS) --filename dataset/$*

#LATENT TRAVERSAL
	
# Metric
# ==============================================================================
metrics: $(foreach p,$(METRICS),$(patsubst %,metric/$p/%.data,$(MODELS)))

metric/gini_index/%.data: models/%/saved_model.pb
	disentangled evaluate $* gini-index $(FLAGS)

metric/mig/%.data: models/%/saved_model.pb
	disentangled evaluate $* mig $(FLAGS)

metric/factorvae_score/%.data: models/%/saved_model.pb
	disentangled evaluate $* factorvae-score $(FLAGS)

# Metric
# ==============================================================================
experiments: $(patsubst %, experiment/%/experiment.complete, $(MODELS))

experiment/%/experiment.complete:
	disentangled experiment $* $(FLAGS)
	
# Other
# ==============================================================================
tensorboard:
	tensorboard --logdir $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)logs --reload_multifile=true $(FLAGS)

clean/logs:
	rm -r -i $(FLAGS) $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)logs
clean/models:
	rm -r -i $(FLAGS) $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)models
clean/images:
	rm -r -i $(FLAGS) $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)images
clean/experiments:
	rm -r -i $(FLAGS) $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)logs/experiment
	rm -r -i $(FLAGS) $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)experiment
	rm -r -i $(FLAGS) $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)models/experiment
