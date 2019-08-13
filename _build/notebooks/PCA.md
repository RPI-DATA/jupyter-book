---
interact_link: content/C:\Users\sjgar\Documents\GitHub\jupyter-book\content\notebooks/pca.ipynb
kernel_name: python3
has_widgets: false
title: 'Principal Component Analysis'
prev_page:
  url: /notebooks/linearregression
  title: 'Linear Regression'
next_page:
  url: /notebooks/timeseries
  title: 'Time Series Analysis'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


<img src="https://raw.githubusercontent.com/RPI-DATA/website/master/static/images/rpilogo.png" alt="RPI LOGO" style="width:400px">

<h1 style="text-align:center">Principal Component Analysis</h1>

<a href="https://colab.research.google.com/github/RPI-DATA/tutorials-intro/blob/master/website/PCA_Introduction.ipynb" target="_blank"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"> </a>



**_Principal Component Analysis (PCA)_** can always be used to simplify data with high dimensions (larger than two) into two-dimensional data by eliminating the least influential features on the data. However, we should know the elimination of data makes the independent variable less interpretable. This notebook was adapted from [amueller's](https://github.com/amueller) notebook, *"1 - PCA"*. Here is the link to his repository: https://github.com/amueller/tutorial_ml_gkbionics.git.

This notebook uses the following pedagogical patterns:
* [**4.10** Coding as translation](https://jupyter4edu.github.io/jupyter-edu-book/catalogue.html#coding-as-translation)
* [**4.16** Now you try (with different data or process)](https://jupyter4edu.github.io/jupyter-edu-book/catalogue.html#now-you-try-with-different-data-or-process)



## Learning Objectives
---
1. How Principal Component Analysis (PCA) works.
2. How PCA can be used to do dimensionality reduction.
3. Understand how PCA deals with the covariance matrix by applying eigenvectors. 



## Eigenvectors
--- 
Before we start to deal with the PCA, we need to first learn how PCA utilizes eigenvectors to gain a diagonalization covariance matrix. Eigenvectors and eigenvalues are the main tools used by PCA to obtain a diagnolization covariance matrix. An eigenvector is a vector whose direction will not be affected by a linear transformation. *Eigenvectors* represents the direction of the largest variance of data, while the corresponding *eigenvalue* decides the magnitude of this variance in those directions.

Here we using a simple (2x2) matrix $A$ to explain it.
$$
A = \begin{bmatrix}
1 & 4 \\
3 & 2 
\end{bmatrix}
$$



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# importing class
import sympy as sp
import numpy as np
import numpy.linalg as lg
A = np.matrix([[1,4],[3,2]])

```
</div>

</div>



In general, the eigenvector $v$ of a matrix $A$ is the vector where the following holds:
$$
Av = \lambda v
$$
for which $\lambda$ stands for the eigenvalue such that linear transformation on $v$ can be defined by $\lambda$

Also, we can solve the equation by:
$$
Av - \lambda v = 0 \\
v(A-\lambda I) = 0
$$
While $I$ is the identity matrix of A 

$$
I = A^TA = AA^T
$$
In this case, if $v$ is none-zero vector than $Det(A - \lambda I) = 0$, since it cannot be invertible, and we can solve $v$ for $A$ depends on this relationship.
$$
I = \begin{bmatrix} 
1 & 0 \\
0 & 1 
\end{bmatrix} \\
$$




<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
def solveLambda(A = A,Lambda = sp.symbols("Lambda", real = True) ):
    I = A*A.I
    I = np.around(I, decimals =0)
    return (A - Lambda*I)
Lambda = sp.symbols("Lambda", real = True)
B = solveLambda(A = A, Lambda = Lambda)
B

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
matrix([[-1.0*Lambda + 1, 4],
        [3, -1.0*Lambda + 2]], dtype=object)
```


</div>
</div>
</div>



$$
(A - \lambda I) = \begin{bmatrix}
1-\lambda & 4 \\
3 & 2 - \lambda 
\end{bmatrix} \\
$$



To solve the $\lambda$ we can use the function *solve* in sympy:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
function = Lambda**2 - 3*Lambda - 10
answer = sp.solve(function, Lambda)
answer

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[-2, 5]
```


</div>
</div>
</div>



In this case, $\lambda_1 = -2$ and $\lambda_2 = 5$, and we can figure out the eigenvectors in two cases.

For $\lambda_1 = -2$



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
identity = np.identity(len(A))
eigenvectors_1 = A - answer[0]*identity
eigenvectors_1

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
matrix([[3.00000000000000, 4],
        [3, 4.00000000000000]], dtype=object)
```


</div>
</div>
</div>



Based on the matrix we can infer the eigenvector can be
$$
v_1 = \begin{bmatrix}
-4 \\
3\end{bmatrix}
$$

For $\lambda = 5$



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
eigenvectors_2 = A - answer[1]*identity
eigenvectors_2 

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
matrix([[-4.00000000000000, 4],
        [3, -3.00000000000000]], dtype=object)
```


</div>
</div>
</div>



Based on the matrix we can infer the eigenvector can be
$$
v_2 = \begin{bmatrix}
1\\
1\end{bmatrix}
$$
All in all, the covariance matrix $A'$ now can be:
$$
A' = v * A \\
$$

Such that we can obtain the matrix $V$
$$
V = \begin{bmatrix}
-4 & 1 \\
3 & 1 
\end{bmatrix}
$$
where $A' = V^{-1} A V$ for the diagnalization:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
V = np.matrix([[-4,1],[3,1]])
diagnalization = V.I * A * V
diagnalization

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
matrix([[-2.00000000e+00,  0.00000000e+00],
        [-1.77635684e-15,  5.00000000e+00]])
```


</div>
</div>
</div>



Hence, the diagonalization covariance matrix is 
$$
\begin{bmatrix}
-2 & 0\\
0 & 5 
\end{bmatrix}
$$
Luckily, PCA can do all of this by applyng the function `pca.fit_transform(x)` and `np.cov()`



## Generating Data
---

To start talking about PCA, we first create 200 random two-dimensional data points and have a look at the raw data.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt
Cov = np.array([[2.9, -2.2], [-2.2, 6.5]])
X = np.random.multivariate_normal([1,2], Cov, size=200)
X  

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[ 1.38383744e+00,  1.04186347e+00],
       [ 5.51503120e-01,  4.59397605e+00],
       [ 2.28938623e+00,  2.47599597e+00],
       [ 2.19188924e-01,  3.76187062e+00],
       [ 1.28817280e+00,  2.26882900e+00],
       [ 1.49215277e+00,  5.43773199e+00],
       [ 2.53729723e+00,  1.08715044e+00],
       [-2.10633441e+00,  5.37888151e+00],
       [ 1.70128797e+00, -3.75436698e+00],
       [-1.21499149e+00,  3.48059163e+00],
       [-7.54648318e-01,  5.62416517e+00],
       [-1.37558065e+00,  2.23346390e+00],
       [ 4.21464198e+00, -2.02410801e-01],
       [ 2.21172328e-01,  5.31123422e+00],
       [ 1.67220400e+00,  1.67759592e+00],
       [-1.08342732e+00,  7.64417741e-01],
       [ 1.07512204e+00,  5.06646797e+00],
       [ 7.51991443e-01,  5.94325847e+00],
       [ 3.27556307e+00, -5.53357218e-01],
       [ 3.86073302e+00,  2.23481202e+00],
       [ 1.08506670e-01,  3.07825464e+00],
       [ 1.51667604e+00, -5.28763254e-01],
       [ 2.11773305e+00,  3.46545163e+00],
       [ 2.38272739e+00,  3.12772008e+00],
       [ 2.43004643e+00,  1.30501739e+00],
       [ 1.47374826e+00,  2.70990735e+00],
       [ 2.83921401e+00, -3.98487101e+00],
       [-9.48501078e-01,  7.36280908e+00],
       [ 1.02124459e+00,  3.21448182e+00],
       [ 3.74240494e+00, -3.80052610e+00],
       [ 3.92856602e+00, -3.58849612e-02],
       [ 1.42662492e+00,  1.63372047e+00],
       [ 1.79334228e+00, -2.36036613e+00],
       [ 4.45535088e+00, -2.86569345e+00],
       [ 3.59338676e+00, -3.45793548e+00],
       [ 8.30053163e-01,  1.49907634e+00],
       [-1.21525957e+00,  2.56792684e-01],
       [ 8.41093256e-01,  1.22547217e+00],
       [-1.69501441e+00,  5.98430597e+00],
       [ 1.43031613e+00,  5.64150340e+00],
       [ 3.44927882e+00,  1.66382944e+00],
       [ 3.81900211e+00, -2.19543036e+00],
       [-3.87124208e+00,  8.29581171e+00],
       [ 4.00111772e+00, -4.27922754e+00],
       [ 3.83101981e+00,  3.29064412e+00],
       [ 9.81171752e-01, -4.99434251e-01],
       [ 1.94119265e+00,  4.67199599e+00],
       [ 1.70492866e+00,  1.30461463e+00],
       [ 4.12720768e+00, -7.26288521e-02],
       [-7.52312808e-01, -1.10538573e+00],
       [ 2.23259800e+00,  5.56937461e+00],
       [ 5.38236269e-01, -1.90110017e-01],
       [ 3.16741171e+00, -1.44677173e+00],
       [-1.27851543e+00,  3.14513557e+00],
       [ 5.59405602e-01,  3.28397792e+00],
       [ 1.04967834e+00,  2.16292436e+00],
       [ 2.07759581e+00,  2.58613039e+00],
       [ 2.49801084e+00, -1.33600160e-01],
       [ 7.01636043e-02,  2.66580633e+00],
       [-8.22980265e-01,  1.66177175e+00],
       [ 1.82036215e+00,  3.64253057e+00],
       [-8.37296475e-01,  6.09361748e+00],
       [ 1.21340867e+00,  3.27451190e+00],
       [ 9.25208337e-01,  4.98364326e-01],
       [ 3.27259833e+00,  1.93455706e+00],
       [ 1.00070636e+00,  1.59008364e+00],
       [-5.68151874e-01, -1.06089282e+00],
       [ 9.09293762e-01,  6.51203314e+00],
       [ 2.47492239e-01,  7.40256013e+00],
       [-9.92062671e-01,  4.70095253e+00],
       [ 1.78360648e-01, -9.52978989e-01],
       [ 4.51547585e-01,  2.35927182e+00],
       [ 1.71056552e+00, -1.67556930e+00],
       [ 2.31695319e+00, -1.44061500e-01],
       [ 1.01846555e+00,  1.88192558e+00],
       [ 7.91568221e-01, -5.55651801e-01],
       [-1.08927425e-01,  6.83735657e+00],
       [ 8.59044230e-01,  3.09460584e+00],
       [ 2.67238279e-01,  3.06640265e+00],
       [ 4.19530508e-01,  1.06701189e+00],
       [-9.94313604e-01,  3.67096177e+00],
       [ 1.32071621e+00, -9.63121475e-01],
       [ 1.89326771e+00,  1.26373806e+00],
       [ 1.66854004e+00,  3.32503622e+00],
       [ 1.34388656e+00,  3.05995813e+00],
       [ 1.63738289e+00,  3.57678427e+00],
       [ 5.55590009e-02,  2.05752678e+00],
       [ 2.97838891e+00, -1.56019610e+00],
       [-1.76374127e+00,  3.41813325e+00],
       [ 3.00685070e+00, -1.77795780e+00],
       [ 6.31432567e-03,  2.44837593e+00],
       [ 2.63884511e+00,  8.79250395e-01],
       [-1.86971890e+00,  2.82835210e+00],
       [-1.65576461e+00,  1.18229879e+00],
       [ 1.14193225e+00,  9.81377514e-01],
       [ 9.67591748e-01,  3.01467398e+00],
       [ 8.50965697e-02,  8.88828385e-01],
       [ 2.44559776e-01,  1.91318437e+00],
       [ 2.36951662e+00, -7.60507571e-01],
       [ 4.05324185e+00, -5.69679189e-01],
       [-1.93261573e-01, -1.06964041e+00],
       [ 2.51804738e+00, -1.45263263e+00],
       [-7.94520085e-01,  2.73488878e+00],
       [-2.87464834e-01,  2.05286405e+00],
       [-8.40100523e-01,  3.48513075e+00],
       [-1.49514859e+00,  1.69273655e+00],
       [-2.58332986e+00,  4.13183244e+00],
       [ 2.54619863e+00,  2.38425639e+00],
       [ 2.98013531e+00, -2.00044713e+00],
       [ 7.21769592e-01, -8.21390714e-01],
       [ 1.27381758e+00, -1.57997158e+00],
       [-1.25560897e-01,  4.23214312e+00],
       [ 3.63441401e+00,  8.44587003e-02],
       [-2.36571393e-01,  3.16966703e+00],
       [ 4.68073279e+00, -7.10871648e-01],
       [ 3.01691079e+00,  1.22995633e+00],
       [-1.64623493e+00,  3.85439166e+00],
       [-1.44897118e+00,  3.70852348e+00],
       [ 1.94901692e+00, -3.12987251e-01],
       [ 1.06230454e+00,  1.33344289e+00],
       [ 2.85633768e+00,  2.55296997e-01],
       [ 1.79762588e+00,  1.01342475e+00],
       [-1.35918547e-01,  2.74876660e+00],
       [-2.19381773e+00,  5.22881102e+00],
       [ 8.27024246e-01,  3.33919600e+00],
       [ 4.40522926e+00,  3.14377449e-01],
       [ 1.50565782e+00,  1.23341347e+00],
       [ 2.19532470e+00,  3.38652837e-01],
       [ 2.63387430e+00,  2.33039580e+00],
       [ 1.57733249e+00,  1.12118265e+00],
       [ 5.29588084e+00, -4.76361410e+00],
       [-1.18380202e+00,  4.88556968e+00],
       [-1.38386784e-01,  4.72498304e+00],
       [ 3.16401438e+00, -7.17221082e-01],
       [-1.05810240e+00,  3.48337849e+00],
       [ 3.11582631e-01,  2.41335816e+00],
       [-1.09330171e+00,  1.73976616e+00],
       [ 1.84467874e+00,  6.75953722e-01],
       [-8.08045806e-01,  1.93311144e+00],
       [ 1.87554417e+00,  2.45738144e+00],
       [-1.93664159e+00,  5.11069364e+00],
       [ 2.07624724e+00, -4.69623682e-01],
       [-6.84527335e-01,  1.58039362e+00],
       [-5.16337981e-01,  3.85267200e+00],
       [ 5.08282993e-01,  5.18143885e+00],
       [-8.66529055e-01,  7.97870195e+00],
       [-1.56776039e+00,  5.96604318e+00],
       [ 1.26252149e+00,  5.07210180e-02],
       [ 3.56861856e-01,  3.98449204e+00],
       [-1.36489579e+00,  1.30205252e+00],
       [-2.66313742e+00,  4.20574961e+00],
       [-5.45195508e-01,  1.35447683e+00],
       [ 4.29338498e-01,  2.54727765e+00],
       [ 5.70758423e-01, -1.09839803e+00],
       [ 1.93716980e+00,  2.76739786e+00],
       [-2.45007695e-01,  5.13053728e+00],
       [ 1.34738911e+00,  9.14006988e+00],
       [ 1.10161001e+00, -3.76250741e+00],
       [-1.88153568e+00,  4.33921926e+00],
       [ 1.61124751e+00,  1.91403407e+00],
       [ 1.34838806e+00,  4.92145174e+00],
       [ 4.84694042e+00, -3.18014218e+00],
       [ 1.06905004e+00,  5.59810374e+00],
       [ 3.25255391e-01,  7.62121687e+00],
       [ 2.15554541e+00,  5.68059855e-01],
       [ 2.67705291e-01,  2.52002657e+00],
       [ 3.91708109e-01,  2.68307605e+00],
       [ 7.67764010e-01,  4.14442882e+00],
       [ 7.48390710e-01,  2.30481020e+00],
       [ 2.59682194e-01,  1.11248540e+00],
       [ 6.94497240e-01,  2.72124348e+00],
       [ 1.55548421e+00, -1.71005566e-01],
       [-1.56193106e+00,  3.62306206e+00],
       [-2.16684028e+00,  5.13772744e+00],
       [-1.82286903e+00,  5.65300036e+00],
       [ 1.88708350e+00, -8.80279146e-01],
       [ 2.35138161e+00,  1.27475964e+00],
       [ 6.36625481e-01,  5.98825455e+00],
       [ 1.33325949e+00,  3.24557416e+00],
       [-5.93547157e-01, -1.06895394e+00],
       [ 2.58666551e+00, -3.89370325e+00],
       [ 2.32124234e+00,  1.94765046e+00],
       [ 3.25631242e+00,  3.45587737e+00],
       [ 7.72193865e-01,  3.27678323e+00],
       [ 5.53229922e-01,  1.71998684e+00],
       [ 1.50981860e+00, -3.44209764e+00],
       [-1.33168718e+00, -7.50108534e-01],
       [-7.25131902e-01,  1.24190386e+00],
       [-1.62072423e+00,  7.67401802e+00],
       [ 1.66222391e+00, -5.34080520e-01],
       [ 6.43929100e-01,  1.09047234e+00],
       [ 1.02770779e+00,  3.72345342e+00],
       [-6.11966234e-01,  1.59675835e+00],
       [ 1.36497945e-01,  2.92899117e-01],
       [ 3.09811395e+00, -5.12086304e-01],
       [ 5.71442092e-01,  2.25865083e+00],
       [ 3.21450663e+00,  3.87686171e+00],
       [ 3.27938842e+00, -9.19335290e-01],
       [-1.42975483e+00,  4.54759266e+00],
       [ 3.02329400e+00,  1.07740589e+00]])
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
np.set_printoptions(4, suppress=True) # show only four decimals
print (X[:10,:])                      # print the first 10 rows of X (from 0 to 9)

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[[ 1.3838  1.0419]
 [ 0.5515  4.594 ]
 [ 2.2894  2.476 ]
 [ 0.2192  3.7619]
 [ 1.2882  2.2688]
 [ 1.4922  5.4377]
 [ 2.5373  1.0872]
 [-2.1063  5.3789]
 [ 1.7013 -3.7544]
 [-1.215   3.4806]]
```
</div>
</div>
</div>



We round the whole data for only 4 decimals.



However, there is no obvious relationship based on this 2-dimensional data, hence we plot it.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plt.figure(figsize=(4,4))
plt.scatter(X[:,0], X[:,1], c= "b", edgecolor = "black")
plt.axis('equal') # equal scaling on both axes

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
(-4.344383174063181, 5.769021935148706, -5.473974894688988, 9.850430677793184)
```


</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](C%3A/Users/sjgar/Documents/GitHub/jupyter-book/_build/images/notebooks/pca_22_1.png)

</div>
</div>
</div>



We can have a look at the actual covariance matrix as well:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print (np.cov(X,rowvar=False))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[[ 3.0059 -2.5038]
 [-2.5038  7.2065]]
```
</div>
</div>
</div>



## Running PCA
---
We would now like to analyze the directions in which the data varies most. For that, we 

1. place the point cloud in the center (0,0) and
2. rotate it, such that the direction with most variance is parallel to the x-axis.

Both steps can be done using PCA, which is conveniently available in sklearn.

We start by loading the PCA class from the sklearn package and creating an instance of the class:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from sklearn.decomposition import PCA
pca = PCA()

```
</div>

</div>



Now, `pca` is an object which has a function `pca.fit_transform(x)` which performs both steps from above to its argument `x`, and returns the centered and rotated version of `x`.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
X_pca = pca.fit_transform(X)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
pca.components_

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[ 0.4227, -0.9063],
       [ 0.9063,  0.4227]])
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
pca.mean_

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([0.946 , 1.9484])
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plt.figure(figsize=(4,4))
plt.scatter(X_pca[:,0], X_pca[:,1],c = "b", edgecolor = "black")
plt.axis('equal');

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](C%3A/Users/sjgar/Documents/GitHub/jupyter-book/_build/images/notebooks/pca_31_0.png)

</div>
</div>
</div>



The covariances between different axes should be zero now. We can double-check by having a look at the non-diagonal entries of the covariance matrix:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print (np.cov(X_pca, rowvar=False))

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[[ 8.3743 -0.    ]
 [-0.      1.8381]]
```
</div>
</div>
</div>



## High-Dimensional Data
---

Our small example above was very easy, since we could get insight into the data by simply plotting it. This approach, however, will not work once you have more than three dimensions. Now, let's use the famous Iris Dataset, which is used to recognize the iris plant and has the following four dimensions:
 * Sepal Length
 * Sepal Width
 * Pedal Length
 * Pedal Width



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from io import open
data = open('bezdekIris.data', 'r').readlines()
iris_HD = np.matrix([np.array(val.split(',')[:4]).astype(float) for val in data[:-1]])

```
</div>

</div>



Let's look at the data again. First, the raw data:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
print (iris_HD[:10])

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[[-0.9007  1.019  -1.3402 -1.3154]
 [-1.143  -0.132  -1.3402 -1.3154]
 [-1.3854  0.3284 -1.3971 -1.3154]
 [-1.5065  0.0982 -1.2834 -1.3154]
 [-1.0218  1.2492 -1.3402 -1.3154]
 [-0.5372  1.9398 -1.1697 -1.0522]
 [-1.5065  0.7888 -1.3402 -1.1838]
 [-1.0218  0.7888 -1.2834 -1.3154]
 [-1.7489 -0.3622 -1.3402 -1.3154]
 [-1.143   0.0982 -1.2834 -1.4471]]
```
</div>
</div>
</div>



Since each dimension has different scale in the Iris Dataset, we can use `StandardScaler` to standardize the data points of all dimensions to have unit variance.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from sklearn.preprocessing import StandardScaler
iris_HD = StandardScaler().fit_transform(iris_HD)
iris_HD

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[-0.9007,  1.019 , -1.3402, -1.3154],
       [-1.143 , -0.132 , -1.3402, -1.3154],
       [-1.3854,  0.3284, -1.3971, -1.3154],
       [-1.5065,  0.0982, -1.2834, -1.3154],
       [-1.0218,  1.2492, -1.3402, -1.3154],
       [-0.5372,  1.9398, -1.1697, -1.0522],
       [-1.5065,  0.7888, -1.3402, -1.1838],
       [-1.0218,  0.7888, -1.2834, -1.3154],
       [-1.7489, -0.3622, -1.3402, -1.3154],
       [-1.143 ,  0.0982, -1.2834, -1.4471],
       [-0.5372,  1.4794, -1.2834, -1.3154],
       [-1.2642,  0.7888, -1.2266, -1.3154],
       [-1.2642, -0.132 , -1.3402, -1.4471],
       [-1.87  , -0.132 , -1.5107, -1.4471],
       [-0.0525,  2.17  , -1.4539, -1.3154],
       [-0.1737,  3.0908, -1.2834, -1.0522],
       [-0.5372,  1.9398, -1.3971, -1.0522],
       [-0.9007,  1.019 , -1.3402, -1.1838],
       [-0.1737,  1.7096, -1.1697, -1.1838],
       [-0.9007,  1.7096, -1.2834, -1.1838],
       [-0.5372,  0.7888, -1.1697, -1.3154],
       [-0.9007,  1.4794, -1.2834, -1.0522],
       [-1.5065,  1.2492, -1.5676, -1.3154],
       [-0.9007,  0.5586, -1.1697, -0.9205],
       [-1.2642,  0.7888, -1.056 , -1.3154],
       [-1.0218, -0.132 , -1.2266, -1.3154],
       [-1.0218,  0.7888, -1.2266, -1.0522],
       [-0.7795,  1.019 , -1.2834, -1.3154],
       [-0.7795,  0.7888, -1.3402, -1.3154],
       [-1.3854,  0.3284, -1.2266, -1.3154],
       [-1.2642,  0.0982, -1.2266, -1.3154],
       [-0.5372,  0.7888, -1.2834, -1.0522],
       [-0.7795,  2.4002, -1.2834, -1.4471],
       [-0.416 ,  2.6304, -1.3402, -1.3154],
       [-1.143 ,  0.0982, -1.2834, -1.3154],
       [-1.0218,  0.3284, -1.4539, -1.3154],
       [-0.416 ,  1.019 , -1.3971, -1.3154],
       [-1.143 ,  1.2492, -1.3402, -1.4471],
       [-1.7489, -0.132 , -1.3971, -1.3154],
       [-0.9007,  0.7888, -1.2834, -1.3154],
       [-1.0218,  1.019 , -1.3971, -1.1838],
       [-1.6277, -1.7434, -1.3971, -1.1838],
       [-1.7489,  0.3284, -1.3971, -1.3154],
       [-1.0218,  1.019 , -1.2266, -0.7889],
       [-0.9007,  1.7096, -1.056 , -1.0522],
       [-1.2642, -0.132 , -1.3402, -1.1838],
       [-0.9007,  1.7096, -1.2266, -1.3154],
       [-1.5065,  0.3284, -1.3402, -1.3154],
       [-0.6583,  1.4794, -1.2834, -1.3154],
       [-1.0218,  0.5586, -1.3402, -1.3154],
       [ 1.4015,  0.3284,  0.5354,  0.2641],
       [ 0.6745,  0.3284,  0.4217,  0.3958],
       [ 1.2803,  0.0982,  0.6491,  0.3958],
       [-0.416 , -1.7434,  0.1375,  0.1325],
       [ 0.7957, -0.5924,  0.4786,  0.3958],
       [-0.1737, -0.5924,  0.4217,  0.1325],
       [ 0.5533,  0.5586,  0.5354,  0.5274],
       [-1.143 , -1.5132, -0.2603, -0.2624],
       [ 0.9168, -0.3622,  0.4786,  0.1325],
       [-0.7795, -0.8226,  0.0807,  0.2641],
       [-1.0218, -2.4339, -0.1466, -0.2624],
       [ 0.0687, -0.132 ,  0.2512,  0.3958],
       [ 0.1898, -1.9736,  0.1375, -0.2624],
       [ 0.311 , -0.3622,  0.5354,  0.2641],
       [-0.2948, -0.3622, -0.0898,  0.1325],
       [ 1.038 ,  0.0982,  0.3649,  0.2641],
       [-0.2948, -0.132 ,  0.4217,  0.3958],
       [-0.0525, -0.8226,  0.1944, -0.2624],
       [ 0.4322, -1.9736,  0.4217,  0.3958],
       [-0.2948, -1.283 ,  0.0807, -0.1308],
       [ 0.0687,  0.3284,  0.5922,  0.7907],
       [ 0.311 , -0.5924,  0.1375,  0.1325],
       [ 0.5533, -1.283 ,  0.6491,  0.3958],
       [ 0.311 , -0.5924,  0.5354,  0.0009],
       [ 0.6745, -0.3622,  0.3081,  0.1325],
       [ 0.9168, -0.132 ,  0.3649,  0.2641],
       [ 1.1592, -0.5924,  0.5922,  0.2641],
       [ 1.038 , -0.132 ,  0.7059,  0.659 ],
       [ 0.1898, -0.3622,  0.4217,  0.3958],
       [-0.1737, -1.0528, -0.1466, -0.2624],
       [-0.416 , -1.5132,  0.0239, -0.1308],
       [-0.416 , -1.5132, -0.033 , -0.2624],
       [-0.0525, -0.8226,  0.0807,  0.0009],
       [ 0.1898, -0.8226,  0.7628,  0.5274],
       [-0.5372, -0.132 ,  0.4217,  0.3958],
       [ 0.1898,  0.7888,  0.4217,  0.5274],
       [ 1.038 ,  0.0982,  0.5354,  0.3958],
       [ 0.5533, -1.7434,  0.3649,  0.1325],
       [-0.2948, -0.132 ,  0.1944,  0.1325],
       [-0.416 , -1.283 ,  0.1375,  0.1325],
       [-0.416 , -1.0528,  0.3649,  0.0009],
       [ 0.311 , -0.132 ,  0.4786,  0.2641],
       [-0.0525, -1.0528,  0.1375,  0.0009],
       [-1.0218, -1.7434, -0.2603, -0.2624],
       [-0.2948, -0.8226,  0.2512,  0.1325],
       [-0.1737, -0.132 ,  0.2512,  0.0009],
       [-0.1737, -0.3622,  0.2512,  0.1325],
       [ 0.4322, -0.3622,  0.3081,  0.1325],
       [-0.9007, -1.283 , -0.4308, -0.1308],
       [-0.1737, -0.5924,  0.1944,  0.1325],
       [ 0.5533,  0.5586,  1.2743,  1.7121],
       [-0.0525, -0.8226,  0.7628,  0.9223],
       [ 1.5227, -0.132 ,  1.2175,  1.1856],
       [ 0.5533, -0.3622,  1.0469,  0.7907],
       [ 0.7957, -0.132 ,  1.1606,  1.3172],
       [ 2.1285, -0.132 ,  1.6153,  1.1856],
       [-1.143 , -1.283 ,  0.4217,  0.659 ],
       [ 1.765 , -0.3622,  1.4448,  0.7907],
       [ 1.038 , -1.283 ,  1.1606,  0.7907],
       [ 1.6438,  1.2492,  1.3311,  1.7121],
       [ 0.7957,  0.3284,  0.7628,  1.0539],
       [ 0.6745, -0.8226,  0.8764,  0.9223],
       [ 1.1592, -0.132 ,  0.9901,  1.1856],
       [-0.1737, -1.283 ,  0.7059,  1.0539],
       [-0.0525, -0.5924,  0.7628,  1.5805],
       [ 0.6745,  0.3284,  0.8764,  1.4488],
       [ 0.7957, -0.132 ,  0.9901,  0.7907],
       [ 2.2497,  1.7096,  1.6722,  1.3172],
       [ 2.2497, -1.0528,  1.7858,  1.4488],
       [ 0.1898, -1.9736,  0.7059,  0.3958],
       [ 1.2803,  0.3284,  1.1038,  1.4488],
       [-0.2948, -0.5924,  0.6491,  1.0539],
       [ 2.2497, -0.5924,  1.6722,  1.0539],
       [ 0.5533, -0.8226,  0.6491,  0.7907],
       [ 1.038 ,  0.5586,  1.1038,  1.1856],
       [ 1.6438,  0.3284,  1.2743,  0.7907],
       [ 0.4322, -0.5924,  0.5922,  0.7907],
       [ 0.311 , -0.132 ,  0.6491,  0.7907],
       [ 0.6745, -0.5924,  1.0469,  1.1856],
       [ 1.6438, -0.132 ,  1.1606,  0.5274],
       [ 1.8862, -0.5924,  1.3311,  0.9223],
       [ 2.492 ,  1.7096,  1.5016,  1.0539],
       [ 0.6745, -0.5924,  1.0469,  1.3172],
       [ 0.5533, -0.5924,  0.7628,  0.3958],
       [ 0.311 , -1.0528,  1.0469,  0.2641],
       [ 2.2497, -0.132 ,  1.3311,  1.4488],
       [ 0.5533,  0.7888,  1.0469,  1.5805],
       [ 0.6745,  0.0982,  0.9901,  0.7907],
       [ 0.1898, -0.132 ,  0.5922,  0.7907],
       [ 1.2803,  0.0982,  0.9333,  1.1856],
       [ 1.038 ,  0.0982,  1.0469,  1.5805],
       [ 1.2803,  0.0982,  0.7628,  1.4488],
       [-0.0525, -0.8226,  0.7628,  0.9223],
       [ 1.1592,  0.3284,  1.2175,  1.4488],
       [ 1.038 ,  0.5586,  1.1038,  1.7121],
       [ 1.038 , -0.132 ,  0.8196,  1.4488],
       [ 0.5533, -1.283 ,  0.7059,  0.9223],
       [ 0.7957, -0.132 ,  0.8196,  1.0539],
       [ 0.4322,  0.7888,  0.9333,  1.4488],
       [ 0.0687, -0.132 ,  0.7628,  0.7907]])
```


</div>
</div>
</div>



We can also try plot a few two-dimensional projections, with combinations of two features at a time:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
colorClass = [val.split(',')[-1].replace('\n', '') for val in data[:-1]]
for i in range(len(colorClass)):
    val = colorClass[i]
    if val == 'Iris-setosa':
        colorClass[i] ='r'
    elif val == 'Iris-versicolor':
        colorClass[i] ='b'
    elif val == 'Iris-virginica':
        colorClass[i] ='g'

plt.figure(figsize=(8,8))
for i in range(0,4):
    for j in range(0,4):
        plt.subplot(4, 4, i * 4 + j + 1)
        plt.scatter(iris_HD[:,i].tolist(), iris_HD[:,j].tolist(),c = colorClass, edgecolors = "black")
        plt.axis('equal')
        plt.gca().set_aspect('equal')

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](C%3A/Users/sjgar/Documents/GitHub/jupyter-book/_build/images/notebooks/pca_41_0.png)

</div>
</div>
</div>



It is not easy to see that this is still a two-dimensional dataset! 

However, if we now do PCA on it, you'll see that the last two dimensions do not matter at all:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
pca = PCA() 
X_HE = pca.fit_transform(iris_HD)
print (X_HE[:10,:])

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[[-2.2647  0.48   -0.1277 -0.0242]
 [-2.081  -0.6741 -0.2346 -0.103 ]
 [-2.3642 -0.3419  0.0442 -0.0284]
 [-2.2994 -0.5974  0.0913  0.066 ]
 [-2.3898  0.6468  0.0157  0.0359]
 [-2.0756  1.4892  0.027  -0.0066]
 [-2.444   0.0476  0.3355  0.0368]
 [-2.2328  0.2231 -0.0887  0.0246]
 [-2.3346 -1.1153  0.1451  0.0269]
 [-2.1843 -0.469  -0.2538  0.0399]]
```
</div>
</div>
</div>



By looking at the data after PCA, it is easy to see the value of last two dimension, especially the last one, is pretty small such that the data can be considered as **still only two-dimensional**. To prove this we can use the code `PCA(0.95)`, which will make the function choose the minimum amount of PCA components such that 95% of the data can be kept.



Let's give it a try!



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
pca = PCA(0.95) 
X_95 = pca.fit_transform(iris_HD)
print (X_95[:10,:])

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[[-2.2647  0.48  ]
 [-2.081  -0.6741]
 [-2.3642 -0.3419]
 [-2.2994 -0.5974]
 [-2.3898  0.6468]
 [-2.0756  1.4892]
 [-2.444   0.0476]
 [-2.2328  0.2231]
 [-2.3346 -1.1153]
 [-2.1843 -0.469 ]]
```
</div>
</div>
</div>



We can see that PCA eliminates the **last two dimensions** because they are redundant for our requirements. Let's plot the two dimensions.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plt.figure(figsize=(4,4))
plt.scatter(X_HE[:,0], X_HE[:,1], c = colorClass, edgecolor = "black")
plt.axis('equal')
plt.gca().set_aspect('equal')

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](C%3A/Users/sjgar/Documents/GitHub/jupyter-book/_build/images/notebooks/pca_48_0.png)

</div>
</div>
</div>



We can have a look on the relationship between each dimention from following plots.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
plt.figure(figsize=(8,8))
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, i * 4 + j + 1)
        plt.scatter(X_HE[:,i], X_HE[:,j], c = colorClass, edgecolor = "black")
        plt.gca().set_xlim(-40,40)
        plt.gca().set_ylim(-40,40)
        plt.axis('equal')
        plt.gca().set_aspect('equal')

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](C%3A/Users/sjgar/Documents/GitHub/jupyter-book/_build/images/notebooks/pca_50_0.png)

</div>
</div>
</div>



It is easy to see that the correlations between the other dimensions (other than the first two) are ambiguous and highly concentrated in either horizontal or vertical lines. This fact suggests that there are large differences between the dimensions we selected, and that the weaker dimensions, by contrast, **can't impact the shape of the graph too much**. 



## Dimension Reduction with PCA
---

We have now seen that there are actually only two useful dimensions in the dataset. Let's throw away even more data - the second dimension - and reconstruct the original data in `D`.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
pca = PCA(1) # only keep one dimension!
X_E = pca.fit_transform(iris_HD)
print (X_E[:10,:])

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
[[-2.2647]
 [-2.081 ]
 [-2.3642]
 [-2.2994]
 [-2.3898]
 [-2.0756]
 [-2.444 ]
 [-2.2328]
 [-2.3346]
 [-2.1843]]
```
</div>
</div>
</div>



Now let's plot the reconstructed data and compare to the original data D. We plot the original data as was, and the reconstruction with only one dimension in purple:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
X_reconstructed = pca.inverse_transform(X_E)
plt.figure(figsize=(8,8))
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, i * 4 + j + 1)
        plt.scatter(iris_HD[:,i].tolist(), iris_HD[:,j].tolist(),c=colorClass, edgecolor = "black")
        plt.scatter(X_reconstructed[:,i], X_reconstructed[:,j],c='purple', edgecolor = "black")
        plt.axis('equal')

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](C%3A/Users/sjgar/Documents/GitHub/jupyter-book/_build/images/notebooks/pca_55_0.png)

</div>
</div>
</div>



## Homework
---
1. Do the PCA reduction on the random 6-dimension data and plot it out.
2. Explan what PCA does on your data.

*The code for data is given below.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
pca=PCA(6)
DATA = np.dot(X,np.random.uniform(0.2,3,(2,6))*(np.random.randint(0,2,(2,6))*2-1))
DATA

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([[  1.831 ,   0.5057,  -3.3554,  -3.5867,   2.2747,   3.6566],
       [ -1.8824,  -5.7901,   1.6444,  -0.1932,   6.005 ,  13.0465],
       [  2.5589,  -0.2421,  -5.0142,  -5.7112,   4.6811,   8.1359],
       ...,
       [  6.4571,   6.0567, -10.3691,  -9.5022,   1.2564,  -0.7318],
       [ -5.4073,  -8.5864,   7.4795,   5.3696,   4.5117,  11.8197],
       [  4.7496,   2.8237,  -8.1859,  -8.1907,   3.5069,   4.6639]])
```


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
## Answer:

pca=PCA(6)
DATA = np.dot(X,np.random.uniform(0.2,3,(2,6))*(np.random.randint(0,2,(2,6))*2-1))
DATA2 = pca.fit_transform(DATA)

plt.figure(figsize=(4,4))
plt.scatter(DATA2[:,0], DATA2[:,1], c = "b", edgecolor = "black")
plt.axis('equal')
plt.gca().set_aspect('equal')

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](C%3A/Users/sjgar/Documents/GitHub/jupyter-book/_build/images/notebooks/pca_58_0.png)

</div>
</div>
</div>



## Conclusion
---
In this notebook we used PCA to a simplify a complex dataset into only the essential dimensions. We saw how eliminating dimensions impacts both the 

**Contributers:** Linghao Dong, Josh Beck, Jose Figueroa, Yuvraj Chopra



<a href="https://colab.research.google.com/github/RPI-DATA/tutorials-intro/blob/master/website/PCA_Introduction.ipynb" target="_blank"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"> </a>


