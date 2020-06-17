<h1>Logistic Regression (逻辑回归)</h1>
<h2>1 模型原理</h2>
<p>逻辑回归使用sigmod函数（ \frac{1}{1 + e^{-x}} ）（如下图）对样本进行回归，之后设定阈值将正负样本分开，实际是一种分类算法。</p>
<p><img src="https://raw.githubusercontent.com/shl5133/shl5133.github.io/master/images/sigmod_function.png" style="width:150px height:250px" /></p>
<p>之所以选用sigmod函数，是由于逻辑回归可以看作利用sigmod函数对后验概率P(y=1 | x)的逼近（具体请见<a href='https://blog.csdn.net/qq_19645269/article/details/79551576'>1</a>）。</p>
<h2>2 问题背景定义</h2>
<p>对于给定的m个样本（X<sub>i</sub>, y<sub>i</sub>）（i \epsilon  [1, m]，X为n维向量）进行二分类：y = 0为负例，y = 1为正例。</p>
<h2>3 公式推导</h2>
<h3>3.1 假设函数（hypothesis function）</h3>
<p>h_{\theta }(x) = \frac{1}{1+e^{-(W^{T}X+b)}} = \frac{1}{1+e^{-\theta ^{T}x}} 	          (1)</p>
<p>注：其中\theta表示模型的参数，即w，b；W^{T}x = w_{1}x_{1} + w_{2}x_{2} + ... + w_{n}x_{n}，\theta^{T}x = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} + ... + \theta_{n}x_{n}；</p>
<h3>3.2 预测正确的概率</h3>
<p>我们将h_{\theta }(x)所给出的结果看作概率（因为sigmod可以将变量从（，-\infty，+\infty）映射到（0，1）之间），即有：</p>
<p>1）p(y = 1 | w, x) = h_{\theta }(x)	        	   (2)</p>
<p>2）p(y = 0 | w, x) = 1 - h_{\theta }(x)	   	   (3)</p>
<p>3）p(correct) = h_{\theta }(x)^{y}(1-h_{\theta }(x))^{1-y}       (4)</p>
<p>注：1) p(correct)表示模型预测正确的概率：当y = 1即样本为正例时，p(correct) = h_{\theta }(x)；当y = 0即样本为负例时，p(correct) = 1 - h_{\theta }(x)。</p>
<p>       2)h_{\theta }(x)的意义：例如对于某患者是否为患病，h_{\theta }(x)输出结果为0.7，则表示患者未患病的概率为70%，患病的概率为30%。</p>
<h3>3.3 最大似然估计</h3>
<p>由于我们的目标是找到合适的\theta使得所有样本的p(correct)最大，即使得模型预测正确所有样本的概率最大，所以我们采用最大似然估计：</p>
<p>L (\theta) = \prod_{i = 1}^{m}p(correct)^{i} = \prod_{i = 1}^{m}h_{\theta }(x^{i})^{y^{i}}(1-h_{\theta }(x^{i}))^{1-y^{i}}							    (5)</p>
<p>连乘不好计算，所以我们对(5)式取对数：</p>
<p>l(\theta) = logL (\theta) = \sum_{i = 1}^{m}log(p(correct)^{i}) = \sum_{i = 1}^{m}(y^{i}log(h_{\theta }(x^{i})) + (1-y^{i})log(1-h_{\theta }(x^{i})))	(6)</p>
<p>最优化任务时习惯上我们希望得到函数的最小值，所以对(6)式取负，即是求解最小值，并得到我们最终的损失函数（交叉熵损失函数）J(\theta)：</p>
<p>J(\theta) = -l(\theta)	            			 (7)</p>
<h3>3.4 最佳\theta值计算</h3>
<p>我们使用梯度下降计算损失函数的最小值，具体算法见<a href='https://en.wikipedia.org/wiki/Gradient_descent'>2</a>，求解梯度（即损失函数偏导）的公式为：</p>
<p>\frac{\partial J(\theta)}{\partial \theta} = \frac{\partial (-\sum_{i = 1}^{m}(y^{i}log(h_{\theta }(x^{i})) + (1-y^{i})log(1-h_{\theta }(x^{i}))))}{\partial \theta}</p>
<p>         = -\sum_{i = 1}^{m}(y^{i}\frac{1}{h_{\theta }(x^{i})}\frac{\partial h_{\theta }(x^{i})}{\partial \theta} - (1 - y^{i})\frac{1}{1 - h_{\theta }(x^{i})}\frac{\partial h_{\theta }(x^{i})}{\partial \theta})</p>
<p>         = -\sum_{i = 1}^{m}(y^{i}\frac{1}{h_{\theta }(x^{i})} - (1 - y^{i})\frac{1}{1 - h_{\theta }(x^{i})})\frac{\partial \frac{1}{1+e^{-\theta ^{T}x^{i}}}}{\partial \theta}</p>
<p>         = -\sum_{i = 1}^{m}(y^{i}\frac{1}{h_{\theta }(x^{i})} - (1 - y^{i})\frac{1}{1 - h_{\theta }(x^{i})})\frac{x^{i}e^{- \theta ^{T}x^{i}}}{(1 + e^{-\theta ^{T}x^{i}})^{2}}</p>
<p>         = -\sum_{i = 1}^{m}(y^{i}\frac{1}{h_{\theta }(x^{i})} - (1 - y^{i})\frac{1}{1 - h_{\theta }(x^{i})})x^{i}h_{\theta }(x^{i})^{2}(\frac{1}{h_{\theta }(x^{i})}-1)</p>
<p>        = -\sum_{i = 1}^{m}(y^{i}\frac{1}{h_{\theta }(x^{i})} - (1 - y^{i})\frac{1}{1 - h_{\theta }(x^{i})})x^{i}h_{\theta }(x^{i})(1 - h_{\theta }(x^{i}))</p>
<p>        = -\sum_{i = 1}^{m}(y^{i}(1 - h_{\theta }(x^{i})) - (1 - y^{i})h_{\theta }(x^{i}))x^{i}</p>
<p>        = -\sum_{i = 1}^{m}( y^{i} - h_{\theta }(x^{i}))x^{i}</p>
<p>        = \sum_{i = 1}^{m}(h_{\theta }(x^{i}) - y^{i})x^{i}</p>
<p>其形式与最小二乘法梯度一致，区别在于最小二乘法中h_{\theta }(x) = W^{T}x，也即是线性回归。</p>
