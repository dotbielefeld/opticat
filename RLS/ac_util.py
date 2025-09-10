import numpy as np
import torch.nn as nn
import copy
import json
import os

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

from RLS.util.log_setup import TournamentEncoder
from RLS.util.generators.default_point_generator import check_conditionals


@dataclass
class BatchMetrics:
    ppo_loss: List[float] = field(default_factory=list)
    clip_fraction: List[float] = field(default_factory=list)
    kl_divergence: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    gradnorm: List[float] = field(default_factory=list)

    def update(self, ppo_loss_val, clip_fraction_val, kl_divergence_val, entropy_val, gradnorm_val):
        """Update batch-level metrics."""
        self.ppo_loss.append(ppo_loss_val)
        self.clip_fraction.append(clip_fraction_val)
        self.kl_divergence.append(kl_divergence_val)
        self.entropy.append(entropy_val)
        self.gradnorm.append(gradnorm_val)

    def compute_epoch_metrics(self) -> Dict[str, float]:
        """Compute averages for batch metrics."""
        return {
            "epoch_loss": sum(self.ppo_loss) / len(self.ppo_loss) if self.ppo_loss else 0.0,
            "epoch_cf": sum(self.clip_fraction) / len(self.clip_fraction) if self.clip_fraction else 0.0,
            "epoch_kl": sum(self.kl_divergence) / len(self.kl_divergence) if self.kl_divergence else 0.0,
            "epoch_en": sum(self.entropy) / len(self.entropy) if self.entropy else 0.0,
        }

@dataclass
class TrainingMetrics:
    epoch_loss: List[float] = field(default_factory=list)
    epoch_cf: List[float] = field(default_factory=list)
    epoch_kl: List[float] = field(default_factory=list)
    epoch_en: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)

    def update_epoch(self, batch_metrics: BatchMetrics):
        """Update epoch-level metrics from batch metrics."""
        epoch_averages = batch_metrics.compute_epoch_metrics()
        self.epoch_loss.append(epoch_averages["epoch_loss"])
        self.epoch_cf.append(epoch_averages["epoch_cf"])
        self.epoch_kl.append(epoch_averages["epoch_kl"])
        self.epoch_en.append(epoch_averages["epoch_en"])
        self.grad_norms.append(batch_metrics.gradnorm)

    def compute_final_metrics(self) -> Tuple[float, float, float, List[float], float]:
        """Compute final aggregated metrics."""
        avg_loss = sum(self.epoch_loss) / len(self.epoch_loss) if self.epoch_loss else 0.0
        avg_cf = sum(self.epoch_cf) / len(self.epoch_cf) if self.epoch_cf else 0.0
        avg_kl = sum(self.epoch_kl) / len(self.epoch_kl) if self.epoch_kl else 0.0
        avg_en = sum(self.epoch_en) / len(self.epoch_en) if self.epoch_en else 0.0
        return avg_loss, avg_cf, avg_kl, self.grad_norms, avg_en


def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    elif isinstance(layer, nn.LayerNorm):
        nn.init.constant_(layer.weight, 1.0)
        nn.init.constant_(layer.bias, 0.0)


def smooth_curve(values, window_size):
    smoothed_values = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
    return smoothed_values

def make_prediction(instances, scenario, rl_agent, output_file):
    # Extract features and instance identifiers
    features = [scenario.features[i] for i in instances]
    instance_ids = [i for i in instances]

    # Get actions from the RL agent
    actions = rl_agent.get_conf_prod(features)

    # Create configuration dictionary
    final_confs = {k: v for k, v in zip(instance_ids, actions)}

    # Clean configurations and handle conditional violations
    for k, conf in final_confs.items():
        clean_conf = copy.copy(conf.conf)
        cond_vio = check_conditionals(scenario, clean_conf)
        for cv in cond_vio:
            clean_conf.pop(cv, None)
        final_confs[k] = {"conf": clean_conf}

    # Write to the output file
    with open(output_file, 'a') as f:
        json.dump(final_confs, f, indent=4, cls=TournamentEncoder)
        f.write(os.linesep)

def save_to_json(data, filename):
    """Save data to a JSON file."""
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def convert_to_serializable(obj):
    """Recursively convert NumPy types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(i) for i in obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj
