import numpy
import scipy.special
from tensorflow.keras.datasets import mnist

class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化网络，设置输入层，中间层，和输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        # 根据输入的训练数据更新节点链路权重
        '''
        把inputs_list, targets_list转换成numpy支持的二维矩阵
        .T表示做矩阵的转置
        '''
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # 计算信号经过输入层后产生的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 中间层神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层接收来自中间层的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    def query(self, inputs):
        # 根据输入数据计算并输出答案
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算最外层接收到的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs

def _main_():
    # 初始化网络
    '''
    由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
    '''
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.1
    n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 读入训练数据
    (x_train, y_train), (x_test, y_test) = mnist.load_data("mnist.npz")
    train_images = x_train.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255

    # 加入epocs,设定网络的训练循环次数
    epochs = 5
    for i in range(epochs):
        print("第", i, "代")
        for train_data, labels_data in zip(train_images, y_train):
            n.train(train_data, labels_data)

    test_data_file = open("dataset/mnist_test.csv")
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    scores = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_number = int(all_values[0])
        print("该图片对应的数字为:", correct_number)
        # 预处理数字图片
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 让网络判断图片对应的数字
        outputs = n.query(inputs)
        # 找到数值最大的神经元对应的编号
        label = numpy.argmax(outputs)
        print("网络认为图片的数字是：", label)
        if label == correct_number:
            scores.append(1)
        else:
            scores.append(0)
    print(scores)

    # 计算图片判断的成功率
    scores_array = numpy.asarray(scores)
    print("perfermance = ", scores_array.sum() / scores_array.size)
    pass


_main_()