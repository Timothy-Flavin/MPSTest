import time
import argparse
import envpool
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

class EnvPoolSB3Wrapper(VecEnv):
    """
    Adapter to prevent SB3 from wrapping EnvPool in DummyVecEnv.
    Translates EnvPool's batched Gymnasium returns into SB3's expected list-of-dicts format.
    """
    def __init__(self, env, num_envs):
        super().__init__(num_envs, env.observation_space, env.action_space)
        self.env = env
        
    def reset(self):
        obs, _ = self.env.reset()
        return obs
        
    def step_async(self, actions):
        self.actions = actions
        
    def step_wait(self):
        obs, rewards, term, trunc, info = self.env.step(self.actions)
        dones = np.logical_or(term, trunc)
        
        # SB3 strict requirement: infos must be a List[Dict]
        # This adds a minor Python loop overhead but is required for SB3 compatibility
        infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            if dones[i]:
                # SB3 looks for 'terminal_observation' during value bootstrapping
                infos[i]["terminal_observation"] = obs[i]
                
        return obs, rewards, dones, infos
        
    def close(self):
        pass
        
    def get_attr(self, name, indices=None):
        return [None] * self.num_envs
        
    def set_attr(self, name, value, indices=None):
        pass
        
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass
        
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, required=True, help="Identifier for the job")
    args = parser.parse_args()

    num_envs = 64
    
    # 1. Initialize the native EnvPool C++ engine
    raw_env = envpool.make("LunarLander-v2", env_type="gymnasium", num_envs=num_envs)
    
    # 2. Wrap it so SB3 recognizes it as a vectorized environment
    env = EnvPoolSB3Wrapper(raw_env, num_envs)

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    # Buffer size will now correctly calculate as: 64 steps * 64 envs = 4096 frames
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=64,
        batch_size=4096,
        device="cuda",
        verbose=0
    )

    total_timesteps = 2_000_000
    print(f"[Job {args.job_id}] Starting training...")
    
    start_time = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.perf_counter() - start_time

    fps = total_timesteps / elapsed
    print(f"[Job {args.job_id}] Completed. Time: {elapsed:.2f}s | FPS: {fps:.2f}")

if __name__ == "__main__":
    main()