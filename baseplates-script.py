import wandb
run = wandb.init()
artifact = run.use_artifact('aimfg-california/Nigel-Baseplates-2022/model-rosy-meadow-247:v0', type='model')
artifact_dir = artifact.download()