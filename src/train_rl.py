import click
import yaml

from utils import EasyDict, instantiate_from_config
from models.rl_agent.agent import PolicyAgent, PolicyAgentConfig
from PongGame.env import PongEnv


@click.command()
@click.option('--config', type=str, default="config/rl.yaml", show_default=True,
              help='YAML config path')
@click.option('--model', type=str, default="saved", show_default=True,
              help='Path to save model checkpoint')
@click.option('--last-checkpoint', type=str, default=None, help='Resume from checkpoint')
@click.option('--num-envs', type=int, default=5, show_default=True,
              help='Number of parallel Gymnasium envs')
@click.option('--rollout-steps', type=int, default=256, show_default=True,
              help='Steps per env per PPO update')
@click.option('--max-games', type=int, default=None, help='Stop after N completed games')
@click.option('--win-streak', type=int, default=None,
              help='Stop once the agent wins N games in a row (player reaches 4-x first)')
@click.option('--show-plot', is_flag=True, help='Save training loss plot at end')
@click.option('--no-replay', is_flag=True, help='Skip best-episode replay at end')
@click.option('--seed', type=int, default=0, show_default=True)
def main(**kwargs):
    options = EasyDict(kwargs)
    with open(options.config, 'r') as f:
        config = EasyDict(**yaml.safe_load(f))

    policy_agent_config = PolicyAgentConfig(**instantiate_from_config(config.policy_agent))
    sample_env = PongEnv(seed=options.seed)
    agent = PolicyAgent(
        env=sample_env,
        config=policy_agent_config,
        model_path=options.model,
        last_checkpoint=options.last_checkpoint,
    )

    agent.train(
        num_envs=options.num_envs,
        rollout_steps=options.rollout_steps,
        show_plot=options.show_plot,
        max_games=options.max_games,
        replay_best=not options.no_replay,
        base_seed=options.seed,
        win_streak=options.win_streak,
    )


if __name__ == "__main__":
    main()
