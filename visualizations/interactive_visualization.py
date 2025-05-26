#!/usr/bin/env python3
import sys
import os
import matplotlib.pyplot as plt

# ensure the project root is on PYTHONPATH
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from environments.maze_environment       import MazeEnvironment
from agents.model_based_survival_agent   import ModelBasedSurvivalAgent
from utils.schedule_swaps                import schedule_reward_swaps

def run_interactive(env, agent, max_steps, pause=0.2):
    state = env.reset()
    agent.set_position(state)
    plt.ion()
    fig, ax = plt.subplots()

    for t in range(max_steps):
        step_num = t + 1
        ax.clear()
        ax.imshow(env.grid, cmap='binary', interpolation='none')

        # rewards: red=low, green=high
        rp = env.get_reward_positions()
        lv, hv = min(rp.values()), max(rp.values())
        for (r,c),v in rp.items():
            col = 'red' if v==lv else 'green'
            ax.plot(c, r, marker='o', color=col, markersize=12)

        # start (blue), exit (brown), agent (black)
        sr, sc = env.start; ax.plot(sc, sr, marker='s', color='blue',  markersize=12)
        er, ec = env.exit;  ax.plot(ec, er, marker='X', color='brown',markersize=12)
        ar, ac = agent.get_position(); ax.plot(ac, ar, marker='o', color='black',markersize=12)

        rem = max_steps - step_num
        ax.set_title(f"Step {step_num}/{max_steps} (remain {rem})  Reward={agent.get_total_reward():.0f}")
        ax.set_xticks(range(len(env.grid[0]))); ax.set_yticks(range(len(env.grid)))
        ax.set_aspect('equal')

        plt.draw()
        plt.pause(pause)

        action = agent.select_action(env)
        ns, rew, done = env.step(action)
        agent.add_reward(rew)
        agent.update(env, action, rew, ns)
        agent.set_position(ns)

        if done:
            ax.set_title(f"Reached exit at step {env.step_count} — Total Reward={agent.get_total_reward():.0f}")
            plt.draw()
            break

    plt.ioff()
    plt.show()

    collected = env.collected
    print(f"\nStep budget = {max_steps}. Collected: {collected}, Total reward = {agent.get_total_reward():.0f}")
    if len(collected) < 2:
        print("✅ Survival agent survived with only one reward.")
    else:
        print("⚠️ Collected both rewards — adjust budget or swap logic.")

if __name__ == "__main__":
    cfg = os.path.join(project_root, "environments", "maze_configs", "maze1.json")
    env = MazeEnvironment.from_json(cfg)

    # compute minimal needed to get both greedily
    from collections import deque
    def bfs_dist(grid, start, goal):
        q = deque([(start[0],start[1],0)]); seen={start}
        moves=[(-1,0),(1,0),(0,-1),(0,1)]
        while q:
            r,c,d = q.popleft()
            if (r,c)==goal: return d
            for dr,dc in moves:
                nr,nc = r+dr,c+dc
                if (0<=nr<len(grid) and 0<=nc<len(grid[0]) and
                    grid[nr][nc]==0 and (nr,nc) not in seen):
                    seen.add((nr,nc)); q.append((nr,nc,d+1))
        return float('inf')

    # determine budget
    sorted_rewards = sorted(env.initial_rewards, key=lambda x: x[1], reverse=True)
    d1  = bfs_dist(env.grid, env.start,     sorted_rewards[0][0])
    d12 = bfs_dist(env.grid, sorted_rewards[0][0], sorted_rewards[1][0])
    max_steps = d1 + d12 - 1
    print(f"Budget set to {max_steps} (one less than min steps to get both)")

    # schedule swaps
    env.swap_steps = schedule_reward_swaps(
        grid             = env.grid,
        start            = env.start,
        reward_positions = [pos for pos,_ in env.initial_rewards],
        exit             = env.exit,
        total_budget     = max_steps,
        num_swaps        = 4,
        jitter           = 2
    )
    env.swap_prob = 1.0

    agent = ModelBasedSurvivalAgent(step_budget=max_steps)
    run_interactive(env, agent, max_steps=max_steps)
