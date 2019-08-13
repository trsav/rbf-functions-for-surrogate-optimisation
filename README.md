# RBF Functions for Surrogate Optimization
 An Implementation of Radial Basis Functions for means of approximating a function, with applications within surrogate based optimization


<p align='center'>
<img src="https://github.com/TomRSavage/RBF-Functions-For-Surrogate-Optimization/blob/master/RBFFunction1D.gif" width="400"> <img src="https://github.com/TomRSavage/RBF-Functions-For-Surrogate-Optimization/blob/master/RBFFunction2D.gif" width="400"> 
<p/>

## Radial Basis Function Explanation

Radial Basis Functions (RBFs) are used to interpolate data, approximating an original or unknown function. Their applications within surrogate modelling are clear, data is sampled within the function space of an expensive function and an RBF is created with a much 'cheaper' evaluation cost. When functions are highly non-linear it is beneficial that they be optimized using stochastic methods, often requiring lots of function evaluations. This compuationally cheaper interpolation of the original function allows for much faster and more efficient optimization. 

Our end goal is to produce a function that exactly predicts the correct result at a datapoint, and interpolates areas between datapoints with good accuracy. This function will be in the form:
<p align="center"> 
 <a href="https://www.codecogs.com/eqnedit.php?latex=f(\textbf&space;x)=&space;\sum^p_{i=1}w_i\phi(||\textbf&space;x-x^{data}_i||)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(\textbf&space;x)=&space;\sum^p_{i=1}w_i\phi(||\textbf&space;x-x^{data}_i||)" title="f(\textbf x)= \sum^p_{i=1}w_i\phi(||\textbf x-x^{data}_i||)" /></a> 
</p>
Bit complicated at first glance but all will be explained. 

We start with a set of sample data that we wish to interpolate along with each point's respective function evaluation as follows:
<p align="center"> 
<a href="https://www.codecogs.com/eqnedit.php?latex=x_{data}=\begin{bmatrix}&space;x^1_1&&space;x^1_2&space;&&space;x_3^1&space;&...&space;&x^1_n\\&space;x^2_1&&space;x^2_2&space;&&space;x_3^2&space;&...&&space;x^2_n\\&space;x^3_1&x_2^3&space;&x^3_3&space;&&space;...&x^3_n\\&space;\vdots&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots\\&space;x^p_1&x^p_2&x^p_3&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{data}=\begin{bmatrix}&space;x^1_1&&space;x^1_2&space;&&space;x_3^1&space;&...&space;&x^1_n\\&space;x^2_1&&space;x^2_2&space;&&space;x_3^2&space;&...&&space;x^2_n\\&space;x^3_1&x_2^3&space;&x^3_3&space;&&space;...&x^3_n\\&space;\vdots&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots\\&space;x^p_1&x^p_2&x^p_3&space;\end{bmatrix}" title="x_{data}=\begin{bmatrix} x^1_1& x^1_2 & x_3^1 &... &x^1_n\\ x^2_1& x^2_2 & x_3^2 &...& x^2_n\\ x^3_1&x_2^3 &x^3_3 & ...&x^3_n\\ \vdots& \vdots & \vdots & \ddots\\ x^p_1&x^p_2&x^p_3 \end{bmatrix}" /></a> &nbsp &nbsp&nbsp&nbsp&nbsp
<a href="https://www.codecogs.com/eqnedit.php?latex=y_{data}=&space;\begin{bmatrix}&space;y^1\\&space;y^2\\&space;y^3\\&space;\vdots\\&space;y^p\\&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{data}=&space;\begin{bmatrix}&space;y^1\\&space;y^2\\&space;y^3\\&space;\vdots\\&space;y^p\\&space;\end{bmatrix}" title="y_{data}= \begin{bmatrix} y^1\\ y^2\\ y^3\\ \vdots\\ y^p\\ \end{bmatrix}" /></a>
</p>

Where p is the number of data points and n is the number of input dimensions.

With this data we then create the interpolation matrix, defined as the following:
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\Psi=\begin{bmatrix}&space;\phi(||x^1-x^1||)&\phi(||x^1-x^2||)&space;&&space;\phi(||x^1-x^3||)&space;&...&\phi(||x^1-x^p||)&space;\\&space;\phi(||x^2-x^1||)&\phi(||x^2-x^2||)&space;&&space;\phi(||x^2-x^3||)&space;&...&\phi(||x^2-x^p||)\\&space;\phi(||x^3-x^1||)&\phi(||x^3-x^2||)&space;&&space;\phi(||x^3-x^3||)&space;&...&\phi(||x^3-x^p||)&space;\\&space;\vdots&\vdots&space;&\vdots&space;&\ddots&\vdots\\&space;\phi(||x^p-x^1||)&\phi(||x^p-x^2||)&space;&&space;\phi(||x^p-x^3||)&space;&...&\phi(||x^p-x^p||)&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Psi=\begin{bmatrix}&space;\phi(||x^1-x^1||)&\phi(||x^1-x^2||)&space;&&space;\phi(||x^1-x^3||)&space;&...&\phi(||x^1-x^p||)&space;\\&space;\phi(||x^2-x^1||)&\phi(||x^2-x^2||)&space;&&space;\phi(||x^2-x^3||)&space;&...&\phi(||x^2-x^p||)\\&space;\phi(||x^3-x^1||)&\phi(||x^3-x^2||)&space;&&space;\phi(||x^3-x^3||)&space;&...&\phi(||x^3-x^p||)&space;\\&space;\vdots&\vdots&space;&\vdots&space;&\ddots&\vdots\\&space;\phi(||x^p-x^1||)&\phi(||x^p-x^2||)&space;&&space;\phi(||x^p-x^3||)&space;&...&\phi(||x^p-x^p||)&space;\end{bmatrix}" title="\Psi=\begin{bmatrix} \phi(||x^1-x^1||)&\phi(||x^1-x^2||) & \phi(||x^1-x^3||) &...&\phi(||x^1-x^p||) \\ \phi(||x^2-x^1||)&\phi(||x^2-x^2||) & \phi(||x^2-x^3||) &...&\phi(||x^2-x^p||)\\ \phi(||x^3-x^1||)&\phi(||x^3-x^2||) & \phi(||x^3-x^3||) &...&\phi(||x^3-x^p||) \\ \vdots&\vdots &\vdots &\ddots&\vdots\\ \phi(||x^p-x^1||)&\phi(||x^p-x^2||) & \phi(||x^p-x^3||) &...&\phi(||x^p-x^p||) \end{bmatrix}" /></a>
</p>

Where the basis function <a href="https://www.codecogs.com/eqnedit.php?latex=\phi(||x_i-x_j||)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi(||x_i-x_j||)" title="\phi(||x_i-x_j||)" /></a> takes the input coordinates of two sample points, computes the euclidian norm of the two points (distance between them), then takes this distance (now denoted as r) and calculates a gaussian value associated with this distance.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\phi(r)=e^{-\epsilon&space;r^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi(r)=e^{-\epsilon&space;r^2}" title="\phi(r)=e^{-\epsilon r^2}" /></a>
</p>
The standard gaussian function is as follows: 
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\phi(r)=\frac{1}{\sqrt{2\pi&space;\sigma}}e^{\frac{-(r-\mu)^2}{2\sigma&space;^2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi(r)=\frac{1}{\sqrt{2\pi&space;\sigma}}e^{\frac{-(r-\mu)^2}{2\sigma&space;^2}}" title="\phi(r)=\frac{1}{\sqrt{2\pi \sigma}}e^{\frac{-(r-\mu)^2}{2\sigma ^2}}" /></a>
</p>
Indicating that the parameter <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /></a> is proportional to the inverse of the standard deviation squared, thus we can think of <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /></a> as being a sort of length scale. The higher the value, the lower the standard deviation of our gaussian function, and the smaller our length scale is. 
Here is a representation of how changing <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /></a> effects our gaussian kernel. 
<p align="center">
<img src="https://github.com/TomRSavage/RBF-Functions-For-Surrogate-Optimization/blob/master/GaussianGif.gif" width="400">
</p>

It should be noted that the interpolation matrix above is symmetrical, as <a href="https://www.codecogs.com/eqnedit.php?latex=\phi(||x_i-x_j||)=\phi(||x_j-x_i||)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi(||x_i-x_j||)=\phi(||x_j-x_i||)" title="\phi(||x_i-x_j||)=\phi(||x_j-x_i||)" /></a>. Therefore only the upper triangular section needs to be calculated, and the tranpose can be added to form the complete matrix.

We now have a matrix containing data relating to how 'correlated' each datapoint is to another datapoint within function space. This is where <a href="https://www.codecogs.com/eqnedit.php?latex=y_{data}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{data}" title="y_{data}" /></a> comes into play. 
By solving the equation <a href="https://www.codecogs.com/eqnedit.php?latex=\textbf&space;w\Psi=y_{data}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textbf&space;w\Psi=y_{data}" title="\textbf w\Psi=y_{data}" /></a> for <a href="https://www.codecogs.com/eqnedit.php?latex=\textbf&space;w" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textbf&space;w" title="\textbf w" /></a>, we are obtaining p weights that scale the interpolation matrix to fit the test data exactly. This provides some explanation to the original function 
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=f(\textbf&space;x)=&space;\sum^p_{i=1}w_i\phi(||\textbf&space;x-x^{data}_i||)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(\textbf&space;x)=&space;\sum^p_{i=1}w_i\phi(||\textbf&space;x-x^{data}_i||)" title="f(\textbf x)= \sum^p_{i=1}w_i\phi(||\textbf x-x^{data}_i||)" /></a>
</p>
Evaluating a coordinate can be interpreted as summing all of the scaled gaussian functions that have been fit around the test data, providing an interpolation. 
These weights along with our test data are what defines our model. It is this embedding of test data within the actual model that allows RBFs to work so efficiently when approximating processes without noise such as expensive computer process simulations, or compuational fluid dynamic systems. 


### Choice of <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /></a>

As a property of radial basis functions is that they are exactly accurate at each data point, the loss  cannot be calculated without reserving some valuable data or sampling the function space again. Therefore the condition number of the matrix <a href="https://www.codecogs.com/eqnedit.php?latex=\Psi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Psi" title="\Psi" /></a> is used as a measurement of how well the data is fit. With the condition number of a matrix relating to it's stability. 
In this implementation <a href="https://www.codecogs.com/eqnedit.php?latex=\Psi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Psi" title="\Psi" /></a> is calculated multiple times and the conditional number found over a logarithmic scale of <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /></a>. Typically resulting in a graph such as the following:

<p align="center">
<img src="https://github.com/TomRSavage/RBF-Functions-For-Surrogate-Optimization/blob/master/Cond_Num.png" width="400"> 
</p>

An <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /></a> value for the gaussian basis function can be approximately chosen as 'on the edge of ill-conditioning' or roughly where a condition number equals 10^12. Note that if you're thinking that this is a terrible way of choosing a hyperparameter then it absolutely is, it's mainly just as a tool to show what's gong on. There are more advanced methods that I will talk about later as well as in another repository (Kriging Interpolation). 

### Effect of <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /></a> on interpolation 

The following 1D and 2D interpolations provide an intuative look at how changing <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /></a> effects the approximate function. Keeping in mind the higher the value of <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /></a>, the smaller the standard deviation of the gaussian basis function. 

<p align="center">
<img src="https://github.com/TomRSavage/RBF-Functions-For-Surrogate-Optimization/blob/master/RBFFunction1D.gif" width="400"> <img src="https://github.com/TomRSavage/RBF-Functions-For-Surrogate-Optimization/blob/master/RBFFunction2D.gif" width="400"> 
</p>

## Example 

The following is a demonstration of how the Rosenbrock function can be interpolated. Starting out with the original function and sampling 30 times:
<p align="center">
<img src="https://github.com/TomRSavage/RBF-Functions-For-Surrogate-Optimization/blob/master/Rosenbrock60.png" width="400"> 
</p>
Then a graph of condition number against <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /></a> is produced and an appropriate value of <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /></a> is chosen: 

<p align="center">
<img src="https://github.com/TomRSavage/RBF-Functions-For-Surrogate-Optimization/blob/master/Cond_Num.png" width="400"> 
</p>
This value of <a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon" title="\epsilon" /></a> is then used to create the final interpolation matrix and corresponding weights. Resulting in the following approximation.

<p align="center">
<img src="https://github.com/TomRSavage/RBF-Functions-For-Surrogate-Optimization/blob/master/RosenbrockRBF.png" width="400"> 
</p>


### Sources

* **Alexander I. J. Forrester, Andras Sobester and Andy J. Keane**- Engineering Design via Surrogate Modelling (John Wiley & Sons, 2008)
