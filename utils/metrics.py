import torch
import torch.distributions as D
from torch.nn.functional import cross_entropy

from utils.constants import *

def instantiate_gmm(outputs, mask=None):
    """Instantiate a sequence of GMMs with torch.distributions

    outputs (dict): values Shape (batch_size, seq_len)
    mask (tensor) shape (batch_size, seq_len) or indices
    """
    if mask is not None:
        weight, loc, scale = [outputs[k][mask] for k in ['weight', 'loc', 'scale']]  # (num_true_in_mask, num_gaussians)
    else:
        weight, loc, scale = [outputs[k] for k in ['weight', 'loc', 'scale']]

    scale = scale.clamp(min=1e-6)  # (num_true_in_mask, num_gaussians)
    mix = D.Categorical(weight / weight.sum(dim=-1, keepdim=True))
    comp = D.Normal(loc, scale)
    gmm = D.mixture_same_family.MixtureSameFamily(mix, comp)
    return gmm

def compute_gmm_nll(outputs, labels, loss_mask):
    """
    Compute negative log likelihood of temporal labels given predicted GMM.

        Parameters:
            outputs (dict): {key (str): value (FloatTensor) Shape (batch_size, seq_len, ntoken, num_gaussians)}
                where key is one of {"weight", "loc", "scale"}
            labels (FloatTensor):                        Shape (batch_size, seq_len)
            loss_mask (BoolTensor):                      Shape (batch_size, seq_len)

        Returns:
            nll (FloatTensor): Negative log likelihood of shape (,)
    """
    target_gmms = instantiate_gmm(outputs, loss_mask)
    target_labels = labels[loss_mask]
    nll = - target_gmms.log_prob(target_labels)  # (num_true_in_mask, num_gaussians)
    return nll.mean()

def compute_loss(output, target, num_regions):
    is_special = (target['region_id'] < N_SPECIAL_TOKENS)
    region_nll = cross_entropy(output['region_id'][~is_special].reshape(-1, num_regions + N_SPECIAL_TOKENS), target["region_id"][~is_special].reshape(-1))
    travel_nll = compute_gmm_nll(output['travel_time'], target['travel_time'], ~is_special)
    departure_nll = compute_gmm_nll(output['departure_time_of_day'], target['departure_time_of_day'], ~is_special)
    loss = region_nll + travel_nll  + departure_nll
    return loss

def compute_gmm_mode_margin_scores(outputs, labels):
    gmm = instantiate_gmm(outputs)
    ll_score = gmm.log_prob(labels).unsqueeze(-1)
    # anomaly_score = gmm.log_prob(labels).exp()
    
    gmm_component_mode = gmm.component_distribution.mode # for each gmm component, where does mode lie on x_axis
    log_probs_at_modes = gmm.component_distribution.log_prob(gmm_component_mode)
    log_pi = gmm.mixture_distribution.logits.log_softmax(dim=-1)
    log_joint_probs = log_probs_at_modes + log_pi  
    log_margin_probs = log_joint_probs.logsumexp(dim=-1, keepdim=True) 
    
    anomaly_score = (log_margin_probs - ll_score).squeeze(-1)  # [num_valid_visits]
    return anomaly_score
    
def compute_gmm_tail_prob(outputs, labels, eps=1e-8):
    gmm = instantiate_gmm(outputs)
    cdf_values = gmm.cdf(labels)   # [num_valid_visits]
    tail_prob = 2 * torch.minimum(1 - cdf_values, cdf_values)  # [num_valid_visits]Add commentMore actions
    tail_prob = torch.clamp(tail_prob, min=eps, max=1.0)
    
    anomaly_score = -torch.log(tail_prob)  # Clamp tail prob to avoid log(0)
    return anomaly_score

def compute_joint_likelihood_anomaly_scores(output, target):
    region_logits = output['region_id']
    region_targets = target['region_id'].unsqueeze(-1)
    region_scores = torch.softmax(region_logits, dim=-1).gather(dim=-1, index=region_targets).squeeze(-1)  # [B, N, 1]

    travel_targets = target['travel_time']
    travel_gmm = instantiate_gmm(output['travel_time'])
    travel_scores = travel_gmm.log_prob(travel_targets).exp()
    
    departure_gmm = instantiate_gmm(output['departure_time_of_day'])
    departure_targets = target['departure_time_of_day']
    departure_scores = departure_gmm.log_prob(departure_targets).exp()
    
    anomaly_scores = 1 - (region_scores * travel_scores * travel_scores)
    
    return anomaly_scores, region_scores, travel_scores, departure_scores
    
    
def compute_nll_anomaly_scores(output, target):
    region_logits = output['region_id']
    region_targets = target['region_id'].unsqueeze(-1)
    region_log_probs = torch.log_softmax(region_logits, dim=-1)
    region_nll = - region_log_probs.gather(dim=-1, index=region_targets).squeeze(-1)  # [B, N, 1]
    
    travel_gmm = instantiate_gmm(output['travel_time'])  
    travel_nll = - travel_gmm.log_prob(target['travel_time'])
    
    departure_gmm = instantiate_gmm(output['departure_time_of_day'])
    departure_nll = - departure_gmm.log_prob(target['departure_time_of_day'])
    
    region_scores = 0.3 * region_nll
    travel_scores = 0.45 * travel_nll
    departure_scores = 0.25 * departure_nll
    
    anomaly_scores =  region_scores + travel_scores + departure_scores

    return anomaly_scores, region_scores, travel_scores, departure_scores

def compute_tail_prob_anomaly_scores(output, target):
    region_logits = output['region_id']
    region_targets = target['region_id'].unsqueeze(-1)
    region_log_probs = torch.log_softmax(region_logits, dim=-1)
    region_scores = - region_log_probs.gather(dim=-1, index=region_targets).squeeze(-1)  # [B, N, 1]

    travel_scores = compute_gmm_tail_prob(output['travel_time'], target['travel_time'])
    departure_scores = compute_gmm_tail_prob(output['departure_time_of_day'], target['departure_time_of_day'])
    
    region_scores = 0.3 * region_scores
    travel_scores = 0.45 * travel_scores
    departure_scores = 0.25 * departure_scores
    
    anomaly_scores =  region_scores + travel_scores + departure_scores
    return anomaly_scores, region_scores, travel_scores, departure_scores
    
    
    
def compute_relative_anomaly_scores(output, target):
    region_logits = output['region_id']
    region_targets = target['region_id'].unsqueeze(-1)
    region_log_probs = torch.log_softmax(region_logits, dim=-1)
    region_ll = region_log_probs.gather(-1, region_targets)  

    region_topk_largest_ll, _ = torch.topk(region_log_probs, k=3, dim=-1, largest=False)  # [N, 10]
    region_scores =  region_topk_largest_ll - region_ll
    row_mask = (region_scores <= 0).any(dim=-1)  # Identify rows containing at least one zero (region falling among top k predictions have negative or zero score).
    region_scores[row_mask] = 0  # Set all elements in those rows to zero
    region_scores = region_scores.mean(dim=-1)  # Average over the regions
    
    travel_scores = compute_gmm_mode_margin_scores(output['travel_time'], target['travel_time'])
    departure_scores = compute_gmm_mode_margin_scores(output['departure_time_of_day'], target['departure_time_of_day'])
    
    region_scores = 0.3 * region_scores
    travel_scores = 0.45 * travel_scores
    departure_scores = 0.25 * departure_scores
    anomaly_scores =  region_scores + travel_scores + departure_scores
    
    return anomaly_scores, region_scores, travel_scores, departure_scores

def compute_anomaly_scores(output, target, is_anomaly_label, anomaly_type_label, method='joint_likelihood'):
    
    if method == 'joint_likelihood':
        anomaly_scores, region_scores, travel_scores, departure_scores = compute_joint_likelihood_anomaly_scores(output, target)
    elif method == 'nll':
        anomaly_scores, region_scores, travel_scores, departure_scores = compute_nll_anomaly_scores(output, target)
    elif method == 'tail_prob':
        anomaly_scores, region_scores, travel_scores, departure_scores = compute_tail_prob_anomaly_scores(output, target)
    elif method == 'relative':
        anomaly_scores, region_scores, travel_scores, departure_scores = compute_relative_anomaly_scores(output, target)
        
        
    is_valid = target['region_id'] >= N_SPECIAL_TOKENS
    
    # visit-level metrics
    visit_anomaly_scores = anomaly_scores[is_valid]
    visit_region_scores = region_scores[is_valid]
    visit_travel_scores = travel_scores[is_valid]
    visit_departure_scores = departure_scores[is_valid]
    anomaly_labels = is_anomaly_label[is_valid]
    anomaly_types = anomaly_type_label[is_valid]

    # agent-level metrics
    agent_level_anomaly_scores = (anomaly_scores * is_valid).max(dim=-1)[0]
    agent_level_labels =  (is_anomaly_label * is_valid).max(dim=-1)[0]
    agent_idx = (anomaly_scores * is_valid).argmax(dim=-1).unsqueeze(-1)
    agent_level_region_scores = region_scores.gather(dim=1, index=agent_idx).squeeze(-1)
    agent_level_travel_scores = travel_scores.gather(dim=1, index=agent_idx).squeeze(-1)
    agent_level_departure_scores = departure_scores.gather(dim=1, index=agent_idx).squeeze(-1)

    
    return (visit_anomaly_scores, visit_region_scores, visit_travel_scores, visit_departure_scores, anomaly_labels, anomaly_types),\
            (agent_level_anomaly_scores, agent_level_region_scores, agent_level_travel_scores, agent_level_departure_scores, agent_level_labels)    