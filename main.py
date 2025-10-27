from copy import deepcopy

import pandas as pd
import numpy as np
import torch
from torch.optim import Adafactor
from tqdm import tqdm, trange

from modules.model import ICAD
from modules.wandb import WandbLogger
from utils.constants import N_SPECIAL_TOKENS
from utils.metrics import *
from utils.parser import parse_args
from utils.preprocess import *
from utils.visualization import plot_anomaly_score_distribution, plot_validation_scores
from sklearn.metrics import average_precision_score, roc_auc_score

import os

def train(train_loader, model, optimizer):
    """
    Trains the model using the provided optimizer and training data.

    Returns:
        list: A list of float. Each is the loss of an iteration.
    """
    model.train()

    losses = []
    for i, batch in enumerate(train_loader):
        input, target = convert_batch_to_model_io(batch, args.device)

        output = model(input)
        loss = compute_loss(output, target, num_regions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

    return losses
    
def train_model(train_loader, model, optimizer, args, logger):
    """Train model with early stopping."""

    best_train_loss = float('inf')
    best_epoch = 0
    best_state_dict = None
    try:
        for epoch in trange(args.max_num_epochs):

            train_losses = train(train_loader, model, optimizer)

            train_loss = sum(train_losses) / len(train_losses)
            logger.log(f"train_loss", train_loss, epoch)

            if train_loss < best_train_loss: # saving the better model, helpful with final model selection since we did not have validation set on the benchmark dataset.
                best_train_loss = train_loss
                best_state_dict = deepcopy(model.state_dict())
                best_epoch = epoch
                torch.save(best_state_dict, os.path.join(args.model_path, f"{args.run_name}_{epoch}.pth"))
                
    except KeyboardInterrupt:
        print(f"Training interrupted. Best model is at epoch {best_epoch}.")

    return best_state_dict


def evaluate(model, loader, loss_prefix, mode):
    """
    Evaluate the model on the given data loader.

    Args:
        loader (torch.utils.data.DataLoader): The data loader containing the evaluation data.
        loss_prefix (str): The prefix to use for the loss key in the results dictionary.

    Returns:
        list: A list of dictionaries containing the evaluation results for each batch.
    """
    model.eval()
    all_scores = []
    all_labels = []
    all_types = []
    total_loss = []

    region_scores = []
    travels_scores = []
    departure_scores = []

    agent_scores = []
    agent_region_scores = []
    agent_travel_scores = []
    agent_departure_scores = []
    agent_labels = []
    with torch.no_grad():
        for batch in loader:
            input, target = convert_batch_to_model_io(batch, args.device)
            
            output = model(input)
                
            loss = compute_loss(output, target, num_regions)
            total_loss.append(loss.item())

            visit_outcome, agent_outcome  = compute_anomaly_scores(output, target, target['anomaly'], target['anomaly_type'], method=args.scoring_method)

            anomaly_scores, region_anomaly_scores, travel_anomaly_scores, departure_anomaly_scores, anomaly_labels, anomaly_types = visit_outcome
            agent_anomaly_scores, agent_region_anomaly_scores, agent_travel_anomaly_scores, agent_departure_anomaly_scores, agent_anomaly_labels = agent_outcome


            all_scores.append(anomaly_scores.cpu())
            region_scores.append(region_anomaly_scores.cpu())
            travels_scores.append(travel_anomaly_scores.cpu())
            departure_scores.append(departure_anomaly_scores.cpu())
            
            agent_scores.append(agent_anomaly_scores.cpu())
            agent_region_scores.append(agent_region_anomaly_scores.cpu())
            agent_travel_scores.append(agent_travel_anomaly_scores.cpu())
            agent_departure_scores.append(agent_departure_anomaly_scores.cpu())
            agent_labels.append(agent_anomaly_labels.cpu())
            
            all_labels.append(anomaly_labels.reshape(-1).cpu())
            all_types.append(anomaly_types.reshape(-1).cpu())

    all_scores = torch.cat(all_scores).numpy()
    agent_scores = torch.cat(agent_scores).numpy()
    agent_region_scores = torch.cat(agent_region_scores).numpy()
    agent_travel_scores = torch.cat(agent_travel_scores).numpy()
    agent_departure_scores = torch.cat(agent_departure_scores).numpy()

    travels_scores = torch.cat(travels_scores).numpy()
    region_scores = torch.cat(region_scores).numpy()
    departure_scores = torch.cat(departure_scores).numpy()


    all_labels = torch.cat(all_labels).cpu().numpy()
    all_types = torch.cat(all_types).cpu().numpy()
    agent_labels = torch.cat(agent_labels).cpu().numpy()
    

    result_dict = {f'{loss_prefix}_loss': sum(total_loss) / len(total_loss)}

    ap = average_precision_score(all_labels, all_scores)
    auroc = roc_auc_score(all_labels, all_scores)
        
    result_dict.update({
        f'Visit-level Total Anomaly {loss_prefix}_AP': ap,
        f'Visit-level Total Anomaly {loss_prefix}_AUROC': auroc,

        f'Agent-level {loss_prefix}_AP': average_precision_score(agent_labels, agent_scores),
        f'Agent-level {loss_prefix}_AUROC': roc_auc_score(agent_labels, agent_scores),

    })
    
    os.makedirs("./results", exist_ok=True)
    scores_df = pd.DataFrame({
        'total_score': all_scores,
        'region_score': region_scores,
        'travel_score': travels_scores,
        'departure_score': departure_scores,
        'is_anomaly': all_labels,
        'anomaly_type': all_types,
    })
    scores_df.to_csv(f"./results/{mode}_{loss_prefix}_visit_scores.csv", index=False)

    agent_df = pd.DataFrame({
        'agent_score': agent_scores,
        'is_anomaly': agent_labels,
        'region_score': agent_region_scores,
        'travel_score': agent_travel_scores,
        'departure_score': agent_departure_scores,
    })
    agent_df.to_csv(f"./results/{mode}_{loss_prefix}_agent_scores.csv", index=False)

    return result_dict


def test_model(test_loader, state_dict, args):
    """Test the best model given the state_dict."""
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    
    test_scores = evaluate(model, test_loader, "test", mode=args.run_name)

    return test_scores

if __name__ == "__main__":
    args = parse_args()

    train_loader, test_loader, num_regions = load_numosim_dataset(args)
    
    model = ICAD(num_regions + N_SPECIAL_TOKENS).to(args.device)

    logger = WandbLogger(project=args.project_name, is_used=args.use_wandb, name=args.run_name)
    logger.log_hyperparams(vars(args))
    logger.watch_model(model)

    optimizer = Adafactor(model.parameters(), lr = args.lr)
    best_state_dict = train_model(train_loader, model, optimizer, args, logger)
    test_score = test_model(test_loader, best_state_dict, args)
    print(test_score)
