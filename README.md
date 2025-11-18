# Convolutional Neural Nerwork in 1000 lines of bare C++
## No third-party libraries!
## Polymorphism
Layers are subclasses of abstract class `Layer`, with implementing functions `forward()`, `backward()`, `save()`, and `load()`.
Gradient stepping(parameter update) and working out downstream gradient are all implemented in `backward()`.
Loader and Loss are all using polymorphism.
Look at `main.cpp` to get informed how to instantialize a model and train it.
## Tensor Viewing
Features are passed in `vector<vector<DataType>>`.
I've defined the `DataType` as `float` in `public.h`. You can change it to double if you like.
The outside vector is the batch, and the inner is the feature vector.
Features of different layer are in different shape. To get to passing through layer, they are all set to a 1-d vector as return value(`batch_size * viewed_length`).
## Model Saving and Loading
The parameters will be saved after every training epoch.
A model's `save` call will sequencely call every layer's `save`.
Same to `load`, which must be called after the whole model is instantialized.

## 1000行纯粹C++实现卷积神经网络
## 不使用第三方库！
## 多态性
各层是抽象类`Layer`的子类，实现了`forward()`、`backward()`、`save()`和`load()`函数。
梯度下降（参数更新）和下游梯度计算均在`backward()`中实现。
数据加载器和损失函数均使用多态实现。
查看`main.cpp`以了解如何实例化模型并进行训练。
## 张量展平
特征以`vector<vector<DataType>>`的形式传递。
我在`public.h`中将`DataType`定义为`float`。如果愿意，你可以将其改为`double`。
外层向量表示批次，内层向量表示特征。
不同层的特征具有不同的形状。为了能使特征在各层间传递，它们在返回时都被设置为一维向量(`批次大小 * 展平后长度`)。
## 模型的保存与加载
每个训练周期结束后都会保存参数。
调用模型的`save`会依次调用每一层的`save`。
`load`也是如此。注意`load`必须在整个模型实例化之后调用。
