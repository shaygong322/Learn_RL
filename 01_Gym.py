import gym  # 导入 Gym 的 Python 接口环境包

env = gym.make('CartPole-v1', render_mode='human', new_step_api=True)   # 构建实验环境，并设置渲染模式和新API
observation = env.reset()  # 重置一个回合

for _ in range(1000):
    env.render()  # 显示图形界面
    action = env.action_space.sample()  # 从动作空间中随机选取一个动作
    observation, reward, terminated, truncated, info = env.step(action)  # 提交动作，并接收返回值
    print(observation)
    if terminated or truncated:
        observation = env.reset()  # 如果回合终止，则重置环境

env.close()  # 关闭环境
