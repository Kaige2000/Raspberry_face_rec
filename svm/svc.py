import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


class Support_Vector_Machine:
    # ���ӻ�
    def __int__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # training
    def fit(self, data):
        # ���������з���
        self.data = data
        opt_dict = {}

        # �������ֱ��
        rotMatrix = lambda theta: np.array([np.sin(theta), np.sin(theta)])
        thetaStep = np.pi / 10
        for theta in np.arange(0, np.pi, thetaStep):
            transforms = [np.array(rotMatrix(theta))]

        # �����ݼ���ƽװ��һ��list��
        all_data = []
        for yi in self.data:
            for featureset in data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        # �ҵ����ݼ��е����ֵ
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)

        # ���岽��
        step_sizes = [self.max_feature_value*0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]

        # Ѱ��b
        b_rang_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value = 10
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_rang_multiple),
                                   self.max_feature_value*b_rang_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # data �е�ÿ����
                        for i in self.data:
                            # ���е�ÿ��ֵ
                            for xi in self[i]:
                                yi = i
                                # ���ڻ�
                                if not yi*(np.dot(w_t, xi)+b) >= 1:
                                    found_option = False
                                    break
                            if not found_option:
                                break
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                    if w[0] < 0:
                        optimized = True
                    else:
                        w = w - step
                norms = sorted([n for n in opt_dict])
                opt_choice = opt_dict[norms[0]]
                self.w = opt_choice[0]
                self.b = opt_choice[1]
                latest_optimum = opt_choice[0][0]*step*2

    # Ԥ��
    def predit(self, features):
        return


data_dict = {-1: np.array([[1, 8], (2, 3), (3, 6)]),
             1: np.array([[1, -2], [3, -4], [3, 0]])
             }
