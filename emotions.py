import numpy as np#1.24.4
#numpy函数用于数学运算
import argparse
#argparse函数用于解析参数
import matplotlib.pyplot as plt#3.7.5
#用于绘制图像
import cv2
#from tensorflow.keras.models import Sequential#tensorflow==2.11.0，keras==2.11.0
from keras.models import Sequential
#sequential用于线性堆叠网络层，将每一层的输出作为下一层的输入
from keras.layers import Dense, Dropout, Flatten
#导入三个不同的神经网络层
#dense：dense层是全连接层，在全连接层中间每个输入的神经元与输出的神经元相连
#dropout：dropout层是一种正则化技术，用于防止神经网络过拟合，中间会有一个rate参数，作为丢弃值，有助于增加鲁棒性。
#flatten：flatten层用于将多维数组转换为一维数组，因为dense需要接收一维数组的输入
from keras.layers import Conv2D
#conv2d是用于处理二维数据的卷积层
from keras.optimizers import Adam
#调用adam优化器
from keras.layers import MaxPooling2D
#调用最大池化层
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#用于图像预处理和增强
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#设置环境变量TF_CPP_MIN_LOG_LEVEL的值为'2'，它控制TensorFlow的日志输出级别。在这里，将日志级别设置为2意味着只显示错误信息，
#不显示其他级别的日志信息。这样可以减少在控制台上显示的日志信息，提高代码的可读性。
ap = argparse.ArgumentParser()#创建一个 argparse.ArgumentParser() 对象 ap，用于解析命令行参数。
ap.add_argument("--mode",help="train/display")
#使用 add_argument() 方法向 ap 中添加命令行参数。
#--mode 是参数的名称。
#help="train/display" 表示该参数的帮助信息，指明该参数可以接受的值是 "train" 或 "display"。
mode = ap.parse_args().mode
#使用 parse_args() 方法解析命令行参数，并将解析结果存储在 mode 变量中。
#.mode 是访问命令行参数 --mode 的值。


def plot_model_history(model_history):

    fig, axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    #通过 ax 对象的 plot() 方法，指定在第一个子图上进行绘制。循环从1-模型训练历史轮次的长度的整数列
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    #通过 ax 对象的 plot() 方法，指定在第一个子图上进行绘制。循环从1-模型训练历史验证机准确度的长度的整数列
    axs[0].set_title('Model Accuracy')
    #选择主题为模型轮数
    axs[0].set_ylabel('Accuracy')
    #选择训练集准确度的历史数据作为Y轴
    axs[0].set_xlabel('Epoch')
    #选择训练轮次作为x轴
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1))
    #以训练集准确度为横轴刻度的标准
    axs[0].legend(['train', 'val'], loc='best')
    #用于在第一个子图上添加图例。图例标识了曲线的含义，使图形更易于理解
    #axs[0].legend() 是调用 ax 对象的 legend() 方法。它将根据给定的标签列表，在第一个子图上显示图例
    #['train', 'val'] 是图例标签的列表
    #loc='best' 表示图例位置的参数。其中 'best' 表示自动选择最佳位置来放置图例
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    #第二个子图参数
    #横轴数据：表示训练轮数的范围、纵轴数据列表：包含了模型在每个训练轮数上的损失值
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    #横轴数据：表示训练轮数的范围、纵轴数据列表：包含了模型在每个训练轮数上的损失值
    axs[1].set_title('Model Loss')
    #主题：模型损失
    axs[1].set_ylabel('Loss')
    #Y轴为损失函数
    axs[1].set_xlabel('Epoch')
    #X轴为训练轮次
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1))
    #以历史损失函数做为X轴刻度
    axs[1].legend(['train', 'val'], loc='best')
    #用于第二个子图添加图例
    fig.savefig('plot.png')
    #保存图像
    plt.show()
    #显示图像

# Define data generators
train_dir = r'data/train'
#训练集地址（绝对路径）
val_dir = r'data/test'
#验证集地址（绝对路径）

num_train = 28709#训练集数量28709
num_val = 7178#验证集数量7178
batch_size = 64 #批量大小，指一次性传入参数的数量，不同的参数数量会影响最后模型的训练精度，如果该值变大会造成内存溢出

num_epoch = 1000#训练轮次，理论上轮次越高精度越高

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
#ImageDataGenerator是 Keras 提供的用于数据增强和预处理的工具类
#rescale=1./255用于图像缩放，这种图像缩放的操作可以提高模型的训练效果和收敛速度
#类似的还有旋转、平移、翻转等等操作
#而这两个创建的变量为数据生成器对象
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')
#使用train对象创建train_generator
#导入训练集地址
#设置训练集图像大小
#设置批量大小
#设置模型颜色
#设置模型类为categorical
validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')
#与上面相同，设置验证集模型


#搭建模型构建卷积神经网络cnn
model = Sequential()
#这一行初始化了Keras中的Sequential模型。Sequential模型是一系列层的线性堆叠，可以轻松地按顺序添加层。
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
#这一行向模型中添加了一个2D卷积层 (Conv2D)。
#32 表示输出的滤波器数量（即输出的通道数）。
#kernel_size=(3, 3) 指定了卷积核的大小为3x3。
#activation='relu' 设置激活函数为ReLU。
#input_shape=(48, 48, 1) 定义了输入数据的形状。这里 (48, 48, 1) 表示输入图像的尺寸为48x48像素，通道数为1（灰度图像）。
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#这一行向模型中再次添加了一个Conv2D层。
#64 表示输出的滤波器数量。
#kernel_size=(3, 3) 设置卷积核的大小。
#activation='relu' 应用ReLU激活函数。
model.add(MaxPooling2D(pool_size=(2, 2)))
#这一行向模型中添加了一个MaxPooling2D层。
#pool_size=(2, 2) 设置池化窗口的大小为2x2，这将每个空间维度（高度和宽度）的输入减少为原来的一半。
model.add(Dropout(0.25))
#这一行向模型中添加了一个Dropout层。
#0.25 指定了训练过程中随机丢弃输入单元的比例。这里，每次更新时会随机丢弃25%的输入单元，有助于防止过拟合

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#这一行向模型中再次添加了一个Conv2D层。
#128 表示输出的滤波器数量（即输出的通道数）。
#kernel_size=(3, 3) 设置卷积核的大小为3x3。
#activation='relu' 应用ReLU激活函数，引入非线性。
model.add(MaxPooling2D(pool_size=(2, 2)))
#这一行添加了另一个MaxPooling2D层。
#pool_size=(2, 2) 指定了2x2的池化窗口，将输入的每个空间维度（高度和宽度）减少为原来的一半。
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#这一行再次向模型中添加了一个Conv2D层。
#128 再次指定了滤波器的数量。
#kernel_size=(3, 3) 设置卷积核的大小。
#activation='relu' 应用ReLU激活函数。
model.add(MaxPooling2D(pool_size=(2, 2)))
#添加了另一个MaxPooling2D层。
#pool_size=(2, 2) 指定了2x2的池化窗口。
model.add(Dropout(0.25))
#这一行向模型中添加了一个Dropout层。
#0.25 指定了训练过程中随机丢弃输入单元的比例。

model.add(Flatten())
#这一行向模型中添加了一个Flatten层。
#Flatten 层用于将输入的二维矩阵表示（在这种情况下，是卷积层的输出）转换为一维向量。它准备数据输入后续的全连接Dense层。
model.add(Dense(1024, activation='relu'))
#这一行向模型中添加了一个Dense（全连接）层。
#1024 指定了层中神经元（单元）的数量。
#activation='relu' 设置激活函数为ReLU，引入非线性。
model.add(Dropout(0.5))
#这一行向模型中添加了一个Dropout层。
#0.5 指定了训练过程中随机丢弃输入单元的比例。这里，每次更新时会随机丢弃50%的输入单元，有助于防止过拟合。
model.add(Dense(7, activation='softmax'))
#这一行向模型中再次添加了一个Dense（全连接）层。
#7 指定了输出神经元的数量，对应于分类任务中的类别数。
#activation='softmax' 应用softmax激活函数，通常用于多类别分类任务。它输出各个类别的概率分布。

#选择模式
if mode == "train":
    #如果模型选择为训练模式
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    #model.compile用来编译模型
    #loss='categorical_crossentropy: 用于衡量预测概率分布与实际标签分布之间的差异。
    #categorical_crossentropy： 通常用于多类别分类问题，其中每个示例属于单个类别，且类别是互斥的。
    #optimizer=Adam(lr=0.0001, decay=1e-6): 这个参数定义了用于训练模型的优化器。
    #在这里，使用的是 Adam 优化器（Adam 优化器根据计算得到的梯度调整模型的权重）。
    #lr=0.0001: 指定学习率，控制优化过程中的步长大小。
    #decay=1e-6: 指定每次更新时学习率的衰减量，这有助于提高训练的收敛性。
    #metrics=['accuracy']: 这个参数指定在训练和测试过程中要监控的指标。在这里，使用 'accuracy'，用于评估模型正确预测类标签的能力。
    model_info = model.fit(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)
    #使用数据生成器validation_generator、train_generator训练keras模型model
    #其中train_generator用于训练的数据生成器。它会实时生成训练数据的批次
    #steps_per_epoch:在宣布一个 epoch 完成之前，从 train_generator 中生成的步数（样本批次的数量）。
    #num_train // batch_size 确定了每个 epoch 的批次数。
    #epochs: 模型的训练轮数。一个 epoch 表示对整个训练数据的一次迭代。
    #validation_data: 这是用于验证的数据生成器。它会生成验证数据的批次。
    #validation_steps: 在每个 epoch 结束时，从 validation_generator 中生成的步数（样本批次的数量）。
    #num_val // batch_size 确定了样本批次的数量。
    #model_info: 这个变量保存了训练历史信息。它包含每个 epoch 的损失和指标值，这对于绘制训练曲线和评估模型性能很有用。

    plot_model_history(model_info)
    #用于绘制模型的训练历史情况
    model.save_weights(r'model_1000.h5')
    #模型保存路径（绝对路径）

#否则选择模型模式为显示模式
elif mode == "display":
    model.load_weights(r'model_1000.h5')
    #加载模型的绝对路径
    cv2.ocl.setUseOpenCL(False)
    #调用opencl功能，它可以利用gpu的并行处理能力来加速各种应用，但是这里选择的是关闭，我们也可以选择开启

    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    #对每一种情绪给定一种键值，分为生气、厌恶、恐惧、快乐、自然、悲伤、惊讶

    cap = cv2.VideoCapture(0)
    #开启摄像头，0代表的是usb摄像头
    while True:
        #
        ret, frame = cap.read()
        #捕获视频帧，读取每一帧图像
        #ret：布尔值，表示读取成功
        #frame：读取出来的图像
        if not ret:
            break
        #如果没有返回ret，则退出
        facecasc = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        #加载特征分类器，opencv自带的分类器
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #将图像转换为灰度图像
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        #facecase级联分类器，使用预训练人脸模型进行人脸检测
        #detectMultiScale：用于在图像中检测不同尺寸的人脸
        #scaleFactor=1.3: 用于调整人脸检测过程中的图像尺度，具体来说，它是指在每个缩放尺度上，
        #算法将图像的尺寸缩小或放大的因子。较小的尺度因子可以检测到更小的人脸，但会增加计算成本。
        #minNeighbors=5: 这是用于确定人脸的邻居数量的参数。它指定在人脸检测过程中，一个矩形框
        #周围需要检测到多少个邻近的矩形框才能将其确认为人脸。较高的邻居数量可以过滤掉一些误检测，但也可能导致错过一些真正的人脸。

        #画人脸矩形框
        for (x, y, w, h) in faces:
            #将人脸的x、y、w、h分离
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            #cv2.rectangle：是opencv中专门用于绘制矩形框的库

            #((x, y-50), (x+w, y+h+10)): 这是第二个参数，它是一个元组，包含两个坐标点，定义了矩形的左上角和右下角。
            #第一个坐标点(x, y-50)表示矩形左上角的x坐标和y坐标减去50，
            #第二个坐标点(x+w, y+h+10)表示矩形右下角的x坐标是x加上宽度w，y坐标是y加上高度h再加上10。

    
            roi_gray = gray[y:y + h, x:x + w]
            #提取出感兴趣区域，
            #[y:y + h, x:x + w]: 这是Python中的切片操作，用于从gray图像中提取一个矩形区域。
            #这个矩形区域的左上角坐标是(x, y)，右下角坐标是(x + w, y + h)。其中：
            #y:y + h表示从y行开始到y + h行结束（不包括y + h行），提取行的切片。
            #x:x + w表示从x列开始到x + w列结束（不包括x + w列），提取列的切片。
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            #np.expand_dims: 这是NumPy库中的一个函数，用于增加数组的维度。
            #它接受两个参数：第一个是要扩展维度的数组，第二个是扩展维度的位置（从0开始计数）。

            #cv2.resize(roi_gray, (48, 48)): 这是OpenCV库中的一个函数，用于改变图像的大小。
            #roi_gray是原始的感兴趣区域（ROI），(48, 48)是新的尺寸，表示将图像调整为48x48像素。

            #np.expand_dims(..., -1): 第一次调用np.expand_dims是将cv2.resize的结果扩展一个维度。
            #由于cv2.resize的结果是一个二维数组（灰度图像），扩展维度位置为-1（表示最后一个维度），
            #这将数组从形状(48, 48)变为形状(48, 48, 1)，即增加了一个单维度的通道。

            #np.expand_dims(..., 0): 第二次调用np.expand_dims是将上一步的结果再扩展一个维度，位置为0，
            #这将数组从形状(48, 48, 1)变为形状(1, 48, 48, 1)。这个操作是为了满足某些深度学习框架对输入数据的维度要求，
            #例如TensorFlow和PyTorch通常期望输入数据的维度是(batch_size, height, width, channels)。
            prediction = model.predict(cropped_img)
            #将处理后的图像数据cropped_img输入到训练好的模型model中，获取模型对这张图像的预测结果，
            #并将这个结果存储在变量prediction中。
            maxindex = int(np.argmax(prediction))
            #确定模型预测结果中概率最高的类别的索引，并将这个索引存储在变量maxindex中
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#在frame图像或视频帧上，以坐标(x+20, y-60)为起始点，使用cv2.FONT_HERSHEY_SIMPLEX字体、大小为1、颜色为白色、线条粗细为2，并开启抗锯齿，绘制文本emotion_dict[maxindex]
        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        #展示显示结果
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #判断如果按下Q键，则退出

    cap.release()
    cv2.destroyAllWindows()
    #中断视频流