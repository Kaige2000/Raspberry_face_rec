import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


class Support_Vector_Machine:
    # 可视化
    def __int__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # training
    def fit(self, data):
        # 在整个类中访问
        self.data = data
        opt_dict = {}

        # 穷举所有直线
        rotMatrix = lambda theta: np.array([np.sin(theta), np.sin(theta)])
        thetaStep = np.pi / 10
        for theta in np.arange(0, np.pi, thetaStep):
            transforms = [np.array(rotMatrix(theta))]
        all_data = []
        for yi in self.data:
            for featureset in data[yi]:
                for feature in featureset:
                    all_data.append(feature)


    # 预测
    def predit(self, features):
        return


data_dict = {-1: np.array([[1, 8], (2, 3), (3, 6)]),
             1: np.array([[1, -2], [3, -4], [3, 0]])

             }
