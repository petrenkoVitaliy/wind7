import numpy as np


class Kalman:
    def __init__(self):
        dt = 0.05
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        self.Q = np.array([
            [1.56e-6, 0, 6.25e-5, 0],
            [0, 1.56e-6, 0, 6.25e-5],
            [6.25e-5, 0, 0.0025, 0],
            [0, 6.25e-5, 0, 0.0025]
        ])

        self.R = np.array([
            [4, 0],
            [0, 4]
        ])

        self.P = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 100, 0],
            [0, 0, 0, 100]
        ])

        self.x = np.array([])
        self.xpred = np.array([])
        self.Ppred = np.copy(self.P)
        self.step = 0

    def init_state(self, initial_position):
        self.x = np.array([
            [initial_position[0][0]],
            [initial_position[1][0]],
            [0.0],
            [0.0]
        ])
        self.step += 1

    def predict(self):
        self.xpred = self.F @ self.x
        self.Ppred = self.F @ self.P @ self.F.T + self.Q

        self.x = self.xpred
        self.P = self.Ppred

    def update(self, results):
        z = np.array(results)
        y = z - self.H @ self.xpred
        S = self.H @ self.Ppred @ self.H.T + self.R
        K = self.Ppred @ self.H.T @ np.linalg.inv(S)

        self.x = self.xpred + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.Ppred

    def next_step(self, results):
        self.step += 1
        print(f"step: {self.step} ----------")
        print(f"results: {results}")

        self.predict()
        print(f"x predict: {self.xpred.T.ravel()}")

        if results is not None:
            self.update(results)
            print(f"x filtered: {self.x.T.ravel()}")


results = [
    [[20], [15]],
    [[25], [40]],
    None,
    None,
    [[45], [70]],
    [[55], [80]]
]

filter = Kalman()
filter.init_state([[10], [10]])

for res in results:
    filter.next_step(res)
