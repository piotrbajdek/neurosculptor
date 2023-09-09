# INTRODUCING NEUROSCULPTOR

![Neurosculptor](https://github.com/piotrbajdek/neurosculptor) is a Lua-based deep learning program distributed under the ![BSD 3-Clause License](https://github.com/piotrbajdek/neurosculptor/blob/main/LICENSE.md). It is a highly configurable classification tool that allows for experimentation with various architectures of dense neural networks in a user-friendly manner.

Unlike its primary use for straightforward classification tasks, this software is primarily developed to tackle complex logical problems that are often beyond human solving capabilities. Its potential applications span across various fields such as science, economics, and more.

Users can experiment with hyperparameter tuning using configuration files. The number of hidden layers can be adjusted from 2 to 5, resulting in a total of 4 to 7 layers. The number of neurons in the input layer and each hidden layer can be freely configured and is virtually limitless. The number of output neurons is set to 1.

In general, 2 hidden layers are suitable for highly nonlinear separation tasks, while 3 hidden layers excel at handling tasks that challenge human capabilities. Architectures with 4 and 5 hidden layers theoretically have the potential to tackle tasks beyond human capabilities but may pose challenges during training.

Neurosculptor v0.1.0 has been verified to work properly with Lua 5.4.4 and LuaJIT 2.1.0-beta3. In terms of speed, LuaJIT is 14 times faster than Lua when benchmarked on a network composed of 250 neurons.

## Configuration files

`learning_rate.conf` - Set a real number.

`iterations.conf` - Specify an integer representing the number of epochs.

`hidden_layers.conf` - Set the number of hidden layers as an integer within the range of 2 to 5.

`hidden_1_size.conf` - Configure the number of neurons in the first hidden layer.

`hidden_2_size.conf` - Configure the number of neurons in the second hidden layer.

`hidden_3_size.conf` - Configure the number of neurons in the third hidden layer.

`hidden_4_size.conf` - Configure the number of neurons in the fourth hidden layer.

`hidden_5_size.conf` - Configure the number of neurons in the fifth hidden layer.

`train_file_x.csv` - This file should contain a binary matrix consisting of the numbers 0 and 1.

`train_file_y.txt` - This file should contain a single column of the numbers 0 and 1, corresponding to the columns in `train_file_x.csv`.

`test_file.csv` - Testing data should be formatted like `train_file_x.csv` but can include any number of lines, with each line representing a separate input for analysis.

Please note that the number of input neurons is automatically calculated based on the data in `train_file_x.csv`, while the number of output neurons is fixed at 1.

# EXAMPLE

![example-1](https://github.com/piotrbajdek/neurosculptor/blob/main/docs/images/example-1.png?raw=true)

`train_file_x.csv` contains a matrix of binary representations for numbers ranging from 0 to 10. This matrix comprises eleven rows and four columns. In contrast, within the `train_file_y.txt`, each row from `train_file_x.csv` is assigned a 1 or a 0, depending on whether the corresponding number is even or odd.

Furthermore, the `test_file.csv` encompasses binary representations of numbers spanning from 11 to 15. As illustrated in the image above, a 3-hidden-layer neural network accurately classifies them as either even or odd, despite these numbers not being present in the training data.
