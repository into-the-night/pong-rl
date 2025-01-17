import click
import yaml

from utils import EasyDict, instantiate_from_config
from models.rl_agent.agent import PolicyAgent, PolicyAgentConfig
from PongGame.env import GameEnvironment

@click.command()
@click.option('--config', help='Config for generation', metavar='YAML', type=str, required=True, default="config/rl.yaml")
@click.option('--model', help='Path to save model', type=str, required=True, default="saved")
@click.option('--dataset', help='Path to save dataset', type=str, required=False, default="dataset")
@click.option('--record', help='Record actions and snapshots', is_flag=True)
@click.option('--clear-dataset', help='Clear dataset folder', is_flag=True, required=False, default=False)
@click.option('--show-plot', help='Show plot', is_flag=True, required=False, default=False)
@click.option('--last-checkpoint', help='Path of checkpoint to resume the training', type=str, required=False)
def main(**kwargs):
    options = EasyDict(kwargs)
    with open(options.config, 'r') as f:
        config = EasyDict(**yaml.safe_load(f))
    env: GameEnvironment = instantiate_from_config(config.env)
    policy_agent_config = PolicyAgentConfig(**instantiate_from_config(config.policy_agent))
    policy_agent = PolicyAgent(env, policy_agent_config, options.model, options.dataset, options.get("last_checkpoint", None))
    policy_agent.train(options.show_plot, options.record, options.clear_dataset)

if __name__ == "__main__":
    main()