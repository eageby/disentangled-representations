train: BetaVAE FactorVAE BetaTCVAE

BetaVAE: train/BetaVAE/Shapes3d 
FactorVAE: train/FactorVAE/Shapes3d 
BetaTCVAE: train/BetaTCVAE/Shapes3d 
SparseVAE: train/SparseVAE/Shapes3d 

train/%:
	disentangled-training --config $@.gin --config mig.gin --gin-parameter 'run_training.iterations=100'
