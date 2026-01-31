import pickle
from src.rl.env import ExecutionEnv
from src.rl.qlearning import qlearning_train


def main():
    Q = 200
    volumes = [10, 40, 200, 100, 100]

    env = ExecutionEnv(
        Q=Q,
        volumes=volumes,
        participation_rate=1.0,
        lambda_impact=1.0,
        lambda_track=20.0,
        terminal_penalty=50000.0,
        q_bin=5,
    )

    Q_table = qlearning_train(env, episodes=10000)

    with open("rl_qtable.pkl", "wb") as f:
        pickle.dump(Q_table, f)

    print("Training done. Saved rl_qtable.pkl")


if __name__ == "__main__":
    main()
