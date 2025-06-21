#!/usr/bin/env python3
import sys, os
import matplotlib.pyplot as plt

# ─── ensure project root is importable ───
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from environments.maze_environment      import MazeEnvironment
from agents.model_based_greedy_agent    import ModelBasedGreedyAgent
from agents.model_based_survival_agent  import ModelBasedSurvivalAgent
from agents.sr_greedy_agent            import SuccessorRepresentationGreedyAgent
from agents.sr_reasonable_agent        import SuccessorRepresentationReasonableAgent
from utils.schedule_swaps               import schedule_fork_swap

def run_interactive(env, agent, pause=0.2):
    state = env.reset()
    agent.set_position(state)
    max_steps = env.max_steps

    plt.ion()
    fig, ax = plt.subplots()

    reached_exit = False

    for t in range(max_steps):
        step_num = t + 1
        ax.clear()
        ax.imshow(env.grid, cmap='binary', interpolation='none')

        # draw rewards
        rp = env.get_reward_positions()
        if rp:
            low, high = min(rp.values()), max(rp.values())
            for (r,c),v in rp.items():
                col = 'red' if v==low else 'green'
                ax.plot(c, r, marker='o', color=col, markersize=12)

        # draw start, exit, agent
        sr, sc = env.start; ax.plot(sc, sr, marker='s', color='blue', markersize=12)
        er, ec = env.exit;  ax.plot(ec, er, marker='X', color='brown', markersize=12)
        ar, ac = agent.get_position(); ax.plot(ac, ar, marker='o', color='black', markersize=12)

        rem = max_steps - step_num
        ax.set_title(f"Step {step_num}/{max_steps} (remain {rem})  Reward={agent.get_total_reward():.0f}")
        ax.set_xticks(range(len(env.grid[0]))); ax.set_yticks(range(len(env.grid)))
        ax.set_aspect('equal')

        plt.draw()
        plt.pause(pause)

        # take a step
        action = agent.select_action(env)
        ns, rew, done = env.step(action)
        agent.add_reward(rew)
        agent.update(env, action, rew, ns)
        agent.set_position(ns)

        if done:
            # distinguish exit vs. budget exhaustion
            if env.agent_pos == env.exit:
                reached_exit = True
                ax.set_title(f"Reached exit in {env.step_count} steps — Total Reward={agent.get_total_reward():.0f}")
            else:
                ax.set_title(f"Out of moves at step {env.step_count} — Total Reward={agent.get_total_reward():.0f}")
            plt.draw()
            break

    plt.ioff()
    plt.show()

    # final report
    print(f"\nStep budget = {env.max_steps}")
    print(f"Collected positions = {env.collected}")
    print(f"Total reward = {agent.get_total_reward():.0f}")
    if reached_exit:
        print("✅ Agent reached the exit.")
    else:
        if isinstance(agent, ModelBasedSurvivalAgent):
            print("❗ Survival agent stopped safely (ran out of moves at budget).")
        else:
            print("⚠️ Agent ran out of moves before reaching the exit.")

if __name__ == "__main__":
    cfg = os.path.join(project_root, "environments", "maze_configs", "maze1.json")
    env = MazeEnvironment.from_json(cfg)
    print("Auto-computed step budget:", env.max_steps)

    # schedule a swap at the decision fork
    env.swap_steps = schedule_fork_swap(
        grid             = env.grid,
        start            = env.start,
        reward_positions = [pos for pos,_ in env.initial_rewards],
        jitter           = 1
    )
    env.swap_prob = 1.0
    print("Fork-scheduled swap step:", env.swap_steps)

    # Choose exactly one agent to run:
    # 1) Impulsive Greedy (ignores budget):
    agent = ModelBasedGreedyAgent(step_budget=env.max_steps)

    # 2) Survival-Aware (won't die if it can avoid it):
    #agent = ModelBasedSurvivalAgent(env.max_steps)

    # 3) SR-Greedy (learns successor representation online, then acts greedily):
    #agent = SuccessorRepresentationGreedyAgent(alpha=0.1)

    # 4) SR-Reasonable (SR + exploration bonus + swap-risk adjustment):
    #agent = SuccessorRepresentationReasonableAgent(alpha=0.1, beta=1.0)

    run_interactive(env, agent)
