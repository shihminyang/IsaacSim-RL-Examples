import argparse
from os.path import join as PJ
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
import torch

from approach_target.callback import TrainLogCallback
from approach_target.env import ApproachTargetEnv
from utils import config, check_stable_baselines_env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Setup
    parser.add_argument("--exp", default="exp1",
                        help="Experiment name")
    parser.add_argument("-H", "--headless", action="store_true",
                        help="Run in headless mode (no GUI)")
    parser.add_argument("-C", "--check_env", action="store_true",
                        help="Check environment")
    parser.add_argument("--root", default="./IsaacSim-RL-Examples")
    args = parser.parse_args()

    ##################################################
    # Setup
    ##################################################
    exp_name = args.exp
    config = config(PJ(args.root, f"configs/{exp_name}.yaml"))
    exp_name = config['exp_name']

    result_dir = config['result_dir']
    log_dir = PJ(result_dir, "logs")
    total_timesteps = config['total_timesteps']

    ##################################################
    # Create simulation and environment
    ##################################################
    if config['rl_env'] == 'ApproachTargetEnv':
        rl_env = ApproachTargetEnv("ApproachTargetEnv",
                                   action_type=config['action_type'],
                                   headless=args.headless, seed=1657)
    else:
        print(f"Environment {config['rl_env']} not support!")
        exit()

    if args.check_env:
        check_stable_baselines_env(rl_env)

    # Parallel training
    if config['num_envs'] > 1:
        rl_env = make_vec_env(lambda: rl_env, n_envs=config['num_envs'])

    ##################################################
    # Model
    ##################################################
    if config['resume']:
        model_path = PJ(result_dir, exp_name, config['model_weight'])
        print(f"Load model from {model_path}")
        model = SAC.load(model_path, env=rl_env, print_system_info=False)
        model.learning_rate = config['learning_rate']
        num_timesteps = int(config['model_weight'].split("_")[1])

    else:
        policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                             net_arch=config['net_arch'])
        lr = config['learning_rate']
        model = SAC('MultiInputPolicy', rl_env,
                    policy_kwargs=policy_kwargs,
                    batch_size=config['batch_size'],
                    learning_rate=lr,
                    device="cuda",
                    tensorboard_log=log_dir)
        num_timesteps = 0

    ##################################################
    # Callback functions
    ##################################################
    checkpoint_callback = CheckpointCallback(
        save_freq=config['save_model_iter'],
        save_path=PJ(result_dir, exp_name),
        name_prefix="ckp",
        verbose=2)
    log_callback = TrainLogCallback(
        config['log_iter'],
        result_dir,
        exp_name,
        num_timesteps=num_timesteps)
    callback_list = [log_callback, checkpoint_callback]

    ##################################################
    # Training
    ##################################################
    model.learn(total_timesteps=total_timesteps, tb_log_name=exp_name,
                reset_num_timesteps=True, callback=callback_list)

    model.save(PJ(result_dir, exp_name, 'complete'))
    rl_env.close()
