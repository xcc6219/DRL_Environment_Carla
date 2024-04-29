import argparse
from env_carla.carla_env import CarlaEnv
from modul import SAC
import numpy as np
import carla


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='Town04')
    parser.add_argument('--task_mode', type=str, default='highway')
    parser.add_argument('--max_time_episode', type=int, default=2000)
    parser.add_argument('--number_of_vehicles', type=int, default=100)
    parser.add_argument('--synchronous_mode', type=bool, default=True)
    parser.add_argument('--no_rendering_mode', type=bool, default=False)
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)
    parser.add_argument('--img_size', type=list, default=[240, 240])
    parser.add_argument('--bev_size', type=list, default=[240, 240])
    parser.add_argument('--pixels_per_meter', type=int, default=3)
    parser.add_argument('--max_past_step', type=int, default=1)
    parser.add_argument('--discrete', type=bool, default=False)
    parser.add_argument('--discrete_acc', type=list, default=[-3.0, 0.0, 3.0])
    parser.add_argument('--discrete_steer', type=list, default=[-0.2, 0.0, 0.2])
    parser.add_argument('--continuous_accel_range', type=float, default=1.0)
    parser.add_argument('--continuous_steer_range', type=float, default=0.3)
    parser.add_argument('--max_speed', type=float, default=34.0)
    parser.add_argument('--out_lane_thred', type=float, default=15.0)
    parser.add_argument('--area', type=list, default=[60.0, 20.0, 40, 40])
    parser.add_argument('--device', type=str, default='cuda')
    # ------------------------------------------------------------------------
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
    parser.add_argument('--gradient_steps', default=1, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=int)
    parser.add_argument('--gamma', default=0.99, type=int)  # discount gamma
    parser.add_argument('--capacity', default=2240, type=int)  # replay buffer size
    parser.add_argument('--max_episode', default=2000, type=int)  # num of  games
    parser.add_argument('--log_interval', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)  # mini batch size
    parser.add_argument('--load', default=False, type=bool)  # load model
    parser.add_argument('--load_dir', default='./SAC_bev/state_4000/', type=str)  # load model
    params = parser.parse_args()
    env = CarlaEnv(opt=params)
    action_dim = env.action_space.shape[0]
    agent = SAC(params)
    ep_r = 0
    if params.mode == 'test':
        agent.load()
        for i in range(params.test_iteration):
            state = env.reset()
            while True:
                #action = agent.select_action(state)
                action=[0,0]
                next_state, reward, done = env.step(np.float32(action))
                ep_r += reward
                if done:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}".format(i, ep_r))
                    ep_r = 0
                    break
                state = next_state
    elif params.mode == 'train':
        if params.load:
            agent.load()
        for i in range(params.max_episode):
            all_rewards = []
            all_speed = []
            state = env.reset()
            while True:
                #action = agent.select_action(state)
                action = [0.05, 0]
                next_state, reward, done = env.step(np.float32(action).reshape(-1))
                all_rewards.append(reward)
                all_speed.append(env.forward_speed)
                agent.replay_buffer.push((state, action, reward, next_state, done))
                state = next_state
                agent.writer.add_scalar('Step/reward', reward, global_step=env.total_step)
                if done:
                    break
            agent.update()
            if i % params.log_interval == 0:
                agent.save()
            total_reward = np.sum(all_rewards)
            average_reward = np.mean(all_rewards)
            std_reward = np.std(all_rewards)
            average_speed = np.mean(all_speed)
            std_speed = np.std(all_speed)
            agent.writer.add_scalar('Episode/total_reward', total_reward, global_step=i)
            agent.writer.add_scalar('Episode/average_reward', average_reward, global_step=i)
            agent.writer.add_scalar('Episode/std_reward', std_reward, global_step=i)
            agent.writer.add_scalar('Episode/average_speed', average_speed, global_step=i)
            agent.writer.add_scalar('Episode/std_speed', std_speed, global_step=i)
            agent.writer.add_scalar('Episode/change_lane', env.lane_change_num, global_step=i)
            agent.writer.add_scalar('Episode/total_distance', env.total_distance, global_step=i)
            agent.writer.add_scalar('Step/Episode_step', env.time_step, global_step=i)
            agent.writer.add_scalar('Step/lane_step', env.lane_change_num/env.time_step, global_step=i)
            print("Step: \t{}.   Episode: \t{}.   Total Reward: \t{:0.2f}.".format(env.time_step, i + 1, total_reward))
    else:
        raise NameError("mode wrong!!!")


if __name__ == '__main__':
    try:
        main()
    finally:
        # 连接到 CARLA 服务器
        client = carla.Client('localhost', 2000)
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        world.tick()
        print("主进程被中断，重置客户端")
