import metaworld
import random
import inspect

ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks

training_envs = []
for name, env_cls in ml10.train_classes.items():
  env = env_cls()
  task = random.choice([task for task in ml10.train_tasks
                        if task.env_name == name])
  env.set_task(task)
  training_envs.append(env)

for env in training_envs:
  obs = env.reset()  # Reset environment
  a = env.action_space.sample()  # Sample an action
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]
  max_action = float(env.action_space.high[0])
  members = inspect.getmembers(env)
  for name, value in members:
    print(f"{name}: {value}")
  # print(state_dim, action_dim, max_action, env.action_space.__dict__)
  obs, reward, done, truncated, info = env.step(a)  # Step the environment with the sampled random action