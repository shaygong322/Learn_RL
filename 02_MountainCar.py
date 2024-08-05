import gym

env = gym.make('MountainCar-v0', new_step_api=True, render_mode='human')  # 构建实验环境，并设置新 API
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('观测范围 = {} ~ {}'.format(env.observation_space.low, env.observation_space.high))
print('动作数 = {}'.format(env.action_space.n))


# 实现智能体来控制小车移动
class SimpleAgent:
    def __init__(self, env):
        pass

    def decide(self, observation):  # 决策
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
                 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action  # 返回动作

    def learn(self, *args):  # 学习
        pass


agent = SimpleAgent(env)


# 让智能体与环境交互
def play(env, agent, render=False, train=False):
    episode_reward = 0.  # 记录回合总奖励，初始值为0
    observation = env.reset()  # 重置游戏环境，开始新回合
    while True:  # 不断循环，直到回合结束
        if render:  # 判断是否显示
            env.render()  # 显示图形界面
        action = agent.decide(observation)
        observation, reward, terminated, truncated, info = env.step(action)  # 执行动作
        episode_reward += reward  # 收集回合奖励
        if train:  # 判断是否训练智能体
            agent.learn(observation, action, reward, terminated)  # 学习
        if terminated or truncated:  # 回合结束，跳出循环
            break
    return episode_reward  # 返回回合总奖励


observation = env.reset(seed=3)  # 设置随机种子，让结果可复现
episode_reward = play(env, agent, render=True)
print('回合奖励 = {}'.format(episode_reward))
env.close()  # 关闭环境
