SHELL := /bin/bash
VPATH := $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)

DATASETS = DSprites Shapes3d 
METHOD = BetaVAE BetaTCVAE FactorVAE BetaSVAE
MODELS := $(foreach p,$(DATASETS),$(patsubst %,%/$p,$(METHOD)))
METRICS := mig gini_index factorvae_score

HYPERPARAMETERS_INDEX = 0 1 2 3 4
RANDOM_SEED_INDEX = 0 1 2

.PHONY: evaluate images metrics train reconstructed examples fixed_factor latents
.PHONY: latent1d latent2d examples fixed_factor concatenate_images metrics experiment
.PHONY: tensorboard clean/logs clean/models clean/experiments

evaluate: images metrics	

# Training
# ==============================================================================
train: $(patsubst %, models/%/saved_model.pb, $(MODELS))
models/%/saved_model.pb:
	disentangled train $* $(FLAGS)

# Images
# ==============================================================================
images: examples fixed_factor reconstructed concatenated

reconstructed: $(patsubst %, images/reconstructed/%.png, $(MODELS))
images/reconstructed/%.png: models/%/saved_model.pb
	disentangled evaluate $* visual $(FLAGS) --no-plot --filename reconstructed/$*

examples: $(patsubst %, images/dataset/%.png, $(DATASETS))
images/dataset/%.png:
	disentangled dataset $* examples  $(FLAGS) --no-plot --filename dataset/$*

fixed_factor: $(patsubst %, images/dataset/fixed_%.png, $(DATASETS))
images/dataset/fixed_%.png:
	disentangled dataset $* fixed $(FLAGS) --no-plot --filename dataset/fixed_$* --verbose \
		> $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)images/dataset/fixed_$*.values

#LATENT TRAVERSAL
latents: latent1d latent2d
latent1d: $(patsubst %, images/latents/%_1d.png, $(MODELS))
latent2d: $(patsubst %, images/latents/%_2d.png, $(MODELS))

images/latents/%_1d.png: models/%/saved_model.pb
	disentangled evaluate $* latent1d $(FLAGS) --no-plot --filename latents/$*_1d

images/latents/%_2d.png: models/%/saved_model.pb
	disentangled evaluate $* latent2d $(FLAGS) --no-plot --filename latents/$*_2d
		
concatenate_images: $(patsubst %, concatenate_images/%, $(DATASETS))
concatenate_images/%: reconstructed
	find $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)images/ -type f -name '*.png' -a -name '*$**' ! -wholename '*reconstructed/$*.png' \
	-exec sh -c 'convert -append $$0 $$@ $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)reconstructed/$*.png' {} + \
	-exec sh -c 'echo $$0 $$@ >  $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)reconstructed/$*.order' {} +  

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
experiments: $(foreach h, $(HYPERPARAMETERS_INDEX), $(foreach r, $(RANDOM_SEED_INDEX), $(patsubst %, experiment/%/HP$h/RS$r/experiment.complete, $(MODELS))))

experiment/%/experiment.complete:
	echo $* | sed -En 's/(.*)\/HP([0-9]+)\/RS([0-9]+)/\1 -h \2 -r \3 --log/p' |  \
		xargs -n6 disentangled experiment

# disentangled experiment $* $(FLAGS)
	
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
