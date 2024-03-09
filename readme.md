# C++ Modular Nielsen Network

C++ implementation of the components necessary to build a neural network using the structure described by Michael Nielsen in his online book, Neural Networks and Deep Learning (http://neuralnetworksanddeeplearning.com/). The goals of this port of the Python code provided by Nielsen are improved speed and a more modular structure; this aims to provide the ability to easily swap parts around and compare their performance without requiring deep re-writes.

## Dependencies

This is a C++ project which requires compilation. The compiler used to develop this program was g++, itself installed through MSYS2 and MinGW-w64 as explained through the following Visual Studio Code tutorial:
- [Using GCC with MinGW](https://code.visualstudio.com/docs/cpp/config-mingw)
It is recommended to follow the same steps and compile the program using g++.

This project relies on the following external libraries:
- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)

## Installation

### 1. Clone the repository

In the command prompt, enter the following commands:

git clone https://github.com/tony-c/cpp-modular-nielsen-network.git
cd cpp-modular-nielsen-network

### 2. Install Eigen

Follow the installation instructions provided on the [official website](http://eigen.tuxfamily.org/dox/GettingStarted.html). It is recommended to place the Eigen folder in the root folder of this project.

### 3. Install g++

Follow the [Visual Studio Code tutorial](https://code.visualstudio.com/docs/cpp/config-mingw) to install g++ through MSYS2 and MinGW-w64. Installing and configuring VSCode is not a necessity to run this project, therefore the steps detailed in the previously-linked tutorial can be safely ignored from the Create a Hello World app section.

### 4. Build the Project

In the command prompt, enter the following commands:

g++ -fdiagnostics-color=always -g ".\network.cpp" -o ".\network.exe" -I "./"

### 5. Run the compiled script

Execute the network.exe program which was created by the previous step.
