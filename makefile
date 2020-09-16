SHELL := /bin/bash
VPATH := $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)

DATASETS = Shapes3d DSprites 
METHOD = BetaVAE BetaTCVAE BetaSVAE FactorVAE 
MODELS := $(foreach p,$(DATASETS),$(patsubst %,%/$p,$(METHOD)))
METRICS := MIG Gini_Index Factorvae_Score DMIG MIG_batch

UNSUPERVISED_DATASETS = CelebA
UNSUPERVISED_MODELS = $(foreach p,$(UNSUPERVISED_DATASETS),$(patsubst %,%/$p,$(METHOD)))

HYPERPARAMETERS_INDEX := 0 1 2 3 4 5
RANDOM_SEED_INDEX := 0 1 2 3 4
RANDOM_SEEDS:= 10 12 25 33 88

.PHONY: evaluate images metrics train reconstructed examples fixed_factor latents
.PHONY: latent1d latent2d examples fixed_factor concatenate_images metrics experiment
.PHONY: tensorboard clean/logs clean/models clean/experiments

evaluate: images metrics	

# Training
# ==============================================================================
train/celeba: $(foreach m, $(METHOD), models/$m/CelebA/saved_model.pb)
models/%/saved_model.pb:
	disentangled train $* $(FLAGS)

# Images
# ==============================================================================
images: examples fixed_factor reconstructed concatenated

reconstructed: $(patsubst %, images/reconstructed/%.png, $(UNSUPERVISED_MODELS)) 

images/reconstructed/%.png: models/%/saved_model.pb
	disentangled evaluate $* visual $(FLAGS) --no-plot --filename reconstructed/$*
	disentangled evaluate $* visual-compare $(FLAGS) --no-plot --filename reconstructed/compare_$*

examples: $(patsubst %, images/dataset/%.png, $(DATASETS))
images/dataset/%.png:
	disentangled dataset $* examples  $(FLAGS) --no-plot --filename dataset/$*

fixed_factor: $(patsubst %, images/dataset/fixed_%.png, $(UNSUPERVISED_DATASETS))
images/dataset/fixed_%.png:
	disentangled dataset $* fixed $(FLAGS) --no-plot --filename dataset/fixed_$* --verbose \
		> $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)images/dataset/fixed_$*.values

#LATENT TRAVERSAL
latents: latent1d latent2d
latent1d: $(patsubst %, images/latents/%_1d.png, $(UNSUPERVISED_MODELS)) 
latent2d: $(patsubst %, images/latents/%_2d.png, $(UNSUPERVISED_MODELS)) 

images/latents/%_1d.png: models/%/saved_model.pb
	disentangled evaluate $* latent1d $(FLAGS) --no-plot --filename latents/$*_1d

images/latents/%_2d.png: models/%/saved_model.pb
	disentangled evaluate $* latent2d $(FLAGS) --no-plot --filename latents/$*_2d
		
# concatenate_images: $(patsubst %, concatenate_images/%, $(DATASETS))
# concatenate_images/%: reconstructed
# 	find $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)images/ -type f -name '*.png' -a -name '*$**' ! -wholename '*reconstructed/$*.png' \
	-exec sh -c 'convert -append $$0 $$@ $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)reconstructed/$*.png' {} + \
	-exec sh -c 'echo $$0 $$@ >  $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)reconstructed/$*.order' {} +  

# Metric
# ==============================================================================
metrics: metrics/gini metrics/mig metrics/mig_batch metrics/dmig metrics/factorvae_score metrics/collapsed
metrics/gini: $(foreach h, $(HYPERPARAMETERS_INDEX), $(foreach r, $(RANDOM_SEEDS), $(patsubst %, logs/experiment/%/HP$h/RS$r/1/eval/Gini_Index/, $(UNSUPERVISED_MODELS))))
metrics/collapsed: $(foreach h, $(HYPERPARAMETERS_INDEX), $(foreach r, $(RANDOM_SEEDS), $(patsubst %, logs/experiment/%/HP$h/RS$r/1/eval/Collapsed/, $(MODELS))))
metrics/mig: $(foreach h, $(HYPERPARAMETERS_INDEX), $(foreach r, $(RANDOM_SEEDS), $(patsubst %, logs/experiment/%/HP$h/RS$r/1/eval/MIG/, $(MODELS))))
metrics/dmig: $(foreach h, $(HYPERPARAMETERS_INDEX), $(foreach r, $(RANDOM_SEEDS), $(patsubst %, logs/experiment/%/HP$h/RS$r/1/eval/DMIG/, $(MODELS))))
metrics/mig_batch: $(foreach h, $(HYPERPARAMETERS_INDEX), $(foreach r, $(RANDOM_SEEDS), $(patsubst %, logs/experiment/%/HP$h/RS$r/1/eval/MIG_batch/, $(MODELS))))
metrics/factorvae_score: $(foreach h, $(HYPERPARAMETERS_INDEX), $(foreach r, $(RANDOM_SEEDS), $(patsubst %, logs/experiment/%/HP$h/RS$r/1/eval/factorvae_Score/, $(MODELS))))

logs/experiment/%/1/eval/Collapsed/: clean/metric/logs/experiment/%/1/eval/Collapsed/
	echo $* | sed -En 's/(\w*\/\w*)\/HP[0-9]+\/RS[0-9]+/\1/p' |  \
	xargs -I {} disentangled evaluate --path $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)/models/experiment/$*/ {} collapsed

logs/experiment/%/1/eval/Gini_Index/: clean/metric/logs/experiment/%/1/eval/Gini_Index/
	echo $* | sed -En 's/(\w*\/\w*)\/HP[0-9]+\/RS[0-9]+/\1/p' |  \
	xargs -I {} disentangled evaluate --path $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)/models/experiment/$*/ {} gini-index

logs/experiment/%/1/eval/MIG/: |clean/metric/logs/experiment/%/1/eval/MIG/
	echo $* | sed -En 's/(\w*\/\w*)\/HP[0-9]+\/RS[0-9]+/\1/p' |  \
	xargs -I {} disentangled evaluate --path $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)/models/experiment/$*/ {} mig

logs/experiment/%/1/eval/MIG_batch/: |clean/metric/logs/experiment/%/1/eval/MIG_batch/
	echo $* | sed -En 's/(\w*\/\w*)\/HP[0-9]+\/RS[0-9]+/\1/p' |  \
	xargs -I {} disentangled evaluate --path $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)/models/experiment/$*/ {} mig-batch

logs/experiment/%/1/eval/DMIG/: |clean/metric/logs/experiment/%/1/eval/DMIG/
	echo $* | sed -En 's/(\w*\/\w*)\/HP[0-9]+\/RS[0-9]+/\1/p' |  \
	xargs -I {} disentangled evaluate --path $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)/models/experiment/$*/ {} dmig

logs/experiment/%/1/eval/factorvae_score/:  |clean/metric/logs/experiment/%/1/eval/factorvae_score/
	echo $* | sed -En 's/(\w*\/\w*)\/HP[0-9]+\/RS[0-9]+/\1/p' |  \
	xargs -I {} disentangled evaluate --path $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)/models/experiment/$*/ {} factorvae-score

clean/metric/%:
	if [ -d "$(DISENTANGLED_REPRESENTATIONS_DIRECTORY)/$*" ]; then rm -r $(DISENTANGLED_REPRESENTATIONS_DIRECTORY)/$*; fi

# Metric
# ==============================================================================
# experiments: $(foreach h, $(HYPERPARAMETERS_INDEX), $(foreach r, $(RANDOM_SEED_INDEX)), $(patsubst %, experiment/%/HP$h/RS$r/experiment.complete, $(MODELS)))

experiment/%/experiment.complete:
	@echo $* | sed -En 's/(.*)\/HP([0-9]+)\/RS([0-9]+)/\1 -h \2 -r \3 --log/p' |  \
		xargs -n6 disentangled experiment

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
