"""
Monitoring functionality for SB3 gym agents - save checkpoints and additional metrics
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np

from stable_baselines3.common.results_plotter import load_results
from stable_baselines3.common.callbacks import BaseCallback

from torch.utils.tensorboard import SummaryWriter

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, save_model_freq: int, log_dir: str, total_timesteps: int, num_envs: int, best_mean="inf", verbose=1):

        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)

        self.check_freq = check_freq           # Check if there is any improvement from the previous best model - frequency 
        self.save_model_freq = save_model_freq # Save model checkpoints at this frequency
        self.model_id = 0
        self.log_dir = log_dir
        self.num_envs = num_envs
        self.save_path = os.path.join(log_dir, 'best_model')  # One best model per run 
        self.save_path_model = os.path.join(log_dir, 'model') # Save many model checkpoints
        self.best_mean_success = 0.0

        # Create path to save models if not exists      #
        # Tensorboard summary writer and custom metrics #
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        custom_logs_path = log_dir + "custom_log_tb/"
        Path(custom_logs_path).mkdir(parents=True, exist_ok=True)

        writer = SummaryWriter(custom_logs_path)
        self.writer = writer

        if (best_mean == "inf"): # Training from the beginning 
            self.best_mean_reward = -np.inf
        else: # Best model already exists and the user has defined a 'best_mean_reward' value
            self.best_mean_reward = float(best_mean)


        self.progress_bar = tqdm(total=total_timesteps, file=sys.stdout)
        self.progress_bar.set_postfix({"Mean Reward": 0.0, "Success ratio": 0.0})

    def _on_step(self) -> bool:
        '''
            Update the best_mean_reward - using the 50 last episodes (adapt if needed)- and save metrics and checkpoints
        '''
        if self.n_calls % self.save_model_freq == 0: # Save model checkpoint
            print("Saving model number: " + str(self.model_id))
            self.model.save(self.save_path_model + "_" + str(self.num_timesteps) + "_" + str(self.model_id))
            self.model_id += 1

        if self.n_calls % self.check_freq == 0: # Update best_mean_reward and metrics

            results = load_results(self.log_dir)

            if len(results.index) > 0:
                mean_reward = np.mean(results["r"].tail(50))
                mean_success = np.mean(results["success"].tail(50))

                # Update summary writer #
                self.writer.add_scalar('Reward/ep_rew_best', self.best_mean_reward, self.num_timesteps)
                self.writer.add_scalar('Reward/ep_reward_mean', mean_reward, self.num_timesteps)
                self.writer.add_scalar('Reward/ep_success_mean', mean_success, self.num_timesteps)
                self.writer.flush()

                self.progress_bar.update(self.n_calls * self.num_envs - self.progress_bar.n)
                self.progress_bar.set_postfix({"Mean Reward": mean_reward, "Success ratio": mean_success})

                if mean_success > self.best_mean_success:
                    self.best_mean_success = mean_success

                # New best model: save it - based on the reward #
                if mean_reward > self.best_mean_reward:
                    print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
                    self.best_mean_reward = mean_reward

        return True

    def _on_training_end(self) -> None:
        """
            Print some stats at the end of the agenttraining
        """

        print("Training ended with best mean reward: " + str(self.best_mean_reward))
        self.writer.add_scalar('best_reward', self.best_mean_reward)
        self.writer.flush()

        self.progress_bar.close()
        self.writer.close()

