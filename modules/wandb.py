import wandb
import os
"Code adapted from https://github.com/USC-InfoLab/NeuroGNN"
class WandbLogger:
    def __init__(self, project, is_used, name=None, entity=None):
        self.is_used = is_used
        if is_used and not name:
            wandb.init(project=project)
        elif is_used and name and not entity:
            wandb.init(project=project, name=name)
        elif is_used and name and entity:
            wandb.init(project=project, name=name, entity=entity)

    def watch_model(self, model):
        if self.is_used:
            wandb.watch(model)

    def log_hyperparams(self, params):
        if self.is_used:
            wandb.config.update(params)

    def log_metrics(self, metrics):
        if self.is_used:
            wandb.log(metrics)

    def log(self, key, value, round_idx):
        if self.is_used:
            wandb.log({key: value}, step=round_idx)

    def log_str(self, key, value):
        if self.is_used:
            wandb.log({key: value})

    def save_file(self, path):
        if path is not None and os.path.exists(path) and self.is_used:
            wandb.save(path)

    def finish(self):
        if self.is_used:
            wandb.finish()