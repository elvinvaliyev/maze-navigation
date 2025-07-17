import csv
import os
import hashlib
import json
import logging

class ExperimentLogger:
    def __init__(self, results_dir, block_metrics_file='block_metrics.csv', transfer_metrics_file='transfer_metrics.csv'):
        self.results_dir = results_dir
        self.block_metrics_path = os.path.join(results_dir, block_metrics_file)
        self.transfer_metrics_path = os.path.join(results_dir, transfer_metrics_file)
        self.block_metrics = []
        self.transfer_metrics = []

    def record_block(self, agent_name, maze_name, reward_name, step_budget, swap_prob, block_index, slope, auc, regret, final_reward):
        self.block_metrics.append({
            'agent': agent_name,
            'maze': maze_name,
            'reward_config': reward_name,
            'step_budget': step_budget,
            'swap_prob': swap_prob,
            'block_index': block_index,
            'slope': slope,
            'auc': auc,
            'regret': regret,
            'final_reward': final_reward
        })

    def record_transfer(self, agent_name, block_index, test_mean, test_std):
        self.transfer_metrics.append({
            'agent': agent_name,
            'block_index': block_index,
            'test_mean': test_mean,
            'test_std': test_std
        })

    def save(self):
        # Save block metrics
        if self.block_metrics:
            with open(self.block_metrics_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.block_metrics[0].keys())
                writer.writeheader()
                writer.writerows(self.block_metrics)
        # Save transfer metrics
        if self.transfer_metrics:
            with open(self.transfer_metrics_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.transfer_metrics[0].keys())
                writer.writeheader()
                writer.writerows(self.transfer_metrics)

def init_logger(config):
    """Initialize logging with a timestamp and config hash in the filename."""
    h = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:6]
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        filename=f"runs_{h}.log",
        level=logging.INFO
    ) 