Title: Live Content

Description: Fetched live

Source: https://cs231n.github.io/optimization-1/

---

<!DOCTYPE html>
<html>

  <head>
  <meta charset="utf-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <title>CS231n Deep Learning for Computer Vision</title>
  <meta name="viewport" content="width=device-width">
  <meta name="description" content="Course materials and notes for Stanford class CS231n: Deep Learning for Computer Vision.">
  <link rel="canonical" href="https://cs231n.github.io/optimization-1/">

  <!-- Custom CSS -->
  <link rel="stylesheet" href="/css/main.css">

  <!-- Google fonts -->
  <link href='https://fonts.googleapis.com/css?family=Roboto:400,300' rel='stylesheet' type='text/css'>

  <!-- Google tracking -->
  <script>
    (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
    (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
    m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
    })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

    ga('create', 'UA-46895817-2', 'auto');
    ga('send', 'pageview');
  </script>
</head>


    <body>

      <script src="https://unpkg.com/vanilla-back-to-top@7.2.1/dist/vanilla-back-to-top.min.js"></script>
      <script>addBackToTop({
        backgroundColor: '#fff',
        innerHTML: 'Back to Top',
        textColor: '#333'
      })</script>
      <style>
        #back-to-top {
          border: 1px solid #ccc;
          border-radius: 0;
          font-family: sans-serif;
          font-size: 14px;
          width: 100px;
          text-align: center;
          line-height: 30px;
          height: 30px;
        }
      </style>

    <header class="site-header">

  <a class="site-title" href="https://cs231n.github.io">CS231n Deep Learning for Computer Vision</a>
  <a class="site-link" href="http://cs231n.stanford.edu/">Course Website</a>

</header>


    <div class="page-content">
      <div class="wrap">
      <div class="post">

  <header class="post-header">
    <h1></h1>
  </header>

  <article class="post-content">
  <p>Table of Contents:</p>

<ul>
  <li><a href="#intro">Introduction</a></li>
  <li><a href="#vis">Visualizing the loss function</a></li>
  <li><a href="#optimization">Optimization</a>
    <ul>
      <li><a href="#opt1">Strategy #1: Random Search</a></li>
      <li><a href="#opt2">Strategy #2: Random Local Search</a></li>
      <li><a href="#opt3">Strategy #3: Following the gradient</a></li>
    </ul>
  </li>
  <li><a href="#gradcompute">Computing the gradient</a>
    <ul>
      <li><a href="#numerical">Numerically with finite differences</a></li>
      <li><a href="#analytic">Analytically with calculus</a></li>
    </ul>
  </li>
  <li><a href="#gd">Gradient descent</a></li>
  <li><a href="#summary">Summary</a></li>
</ul>

<p><a name="intro"></a></p>

<h3 id="introduction">Introduction</h3>

<p>In the previous section we introduced two key components in context of the image classification task:</p>

<ol>
  <li>A (parameterized) <strong>score function</strong> mapping the raw image pixels to class scores (e.g. a linear function)</li>
  <li>A <strong>loss function</strong> that measured the quality of a particular set of parameters based on how well the induced scores agreed with the ground truth labels in the training data. We saw that there are many ways and versions of this (e.g. Softmax/SVM).</li>
</ol>

<p>Concretely, recall that the linear function had the form \( f(x_i, W) =  W x_i \) and the SVM we developed was formulated as:</p>

\[L = \frac{1}{N} \sum_i \sum_{j\neq y_i} \left[ \max(0, f(x_i; W)_j - f(x_i; W)_{y_i} + 1) \right] + \alpha R(W)\]

<p>We saw that a setting of the parameters \(W\) that produced predictions for examples \(x_i\) consistent with their ground truth labels \(y_i\) would also have a very low loss \(L\). We are now going to introduce the third and last key component: <strong>optimization</strong>. Optimization is the process of finding the set of parameters \(W\) that minimize the loss function.</p>

<p><strong>Foreshadowing:</strong> Once we understand how these three core components interact, we will revisit the first component (the parameterized function mapping) and extend it to functions much more complicated than a linear mapping: First entire Neural Networks, and then Convolutional Neural Networks. The loss functions and the optimization process will remain relatively unchanged.</p>

<p><a name="vis"></a></p>

<h3 id="visualizing-the-loss-function">Visualizing the loss function</h3>

<p>The loss functions we’ll look at in this class are usually defined over very high-dimensional spaces (e.g. in CIFAR-10 a linear classifier weight matrix is of size [10 x 3073] for a total of 30,730 parameters), making them difficult to visualize. However, we can still gain some intuitions about one by slicing through the high-dimensional space along rays (1 dimension), or along planes (2 dimensions). For example, we can generate a random weight matrix \(W\) (which corresponds to a single point in the space), then march along a ray and record the loss function value along the way. That is, we can generate a random direction \(W_1\) and compute the loss along this direction by evaluating \(L(W + a W_1)\) for different values of \(a\). This process generates a simple plot with the value of \(a\) as the x-axis and the value of the loss function as the y-axis. We can also carry out the same procedure with two dimensions by evaluating the loss \( L(W + a W_1 + b W_2) \) as we vary \(a, b\). In a plot, \(a, b\) could then correspond to the x-axis and the y-axis, and the value of the loss function can be visualized with a color:</p>

<div class="fig figcenter fighighlight">
  <img src="/assets/svm1d.png" />
  <img src="/assets/svm_one.jpg" />
  <img src="/assets/svm_all.jpg" />
  <div class="figcaption">
    Loss function landscape for the Multiclass SVM (without regularization) for one single example (left,middle) and for a hundred examples (right) in CIFAR-10. Left: one-dimensional loss by only varying <b>a</b>. Middle, Right: two-dimensional loss slice, Blue = low loss, Red = high loss. Notice the piecewise-linear structure of the loss function. The losses for multiple examples are combined with average, so the bowl shape on the right is the average of many piece-wise linear bowls (such as the one in the middle).
  </div>
</div>

<p>We can explain the piecewise-linear structure of the loss function by examining the math. For a single example we have:</p>

\[L_i = \sum_{j\neq y_i} \left[ \max(0, w_j^Tx_i - w_{y_i}^Tx_i + 1) \right]\]

<p>It is clear from the equation that the data loss for each example is a sum of (zero-thresholded due to the \(\max(0,-)\) function) linear functions of \(W\). Moreover, each row of \(W\) (i.e. \(w_j\)) sometimes has a positive sign in front of it (when it corresponds to a wrong class for an example), and sometimes a negative sign (when it corresponds to the correct class for that example). To make this more explicit, consider a simple dataset that contains three 1-dimensional points and three classes. The full SVM loss (without regularization) becomes:</p>

\[\begin{align}
L_0 = &amp; \max(0, w_1^Tx_0 - w_0^Tx_0 + 1) + \max(0, w_2^Tx_0 - w_0^Tx_0 + 1) \\\\
L_1 = &amp; \max(0, w_0^Tx_1 - w_1^Tx_1 + 1) + \max(0, w_2^Tx_1 - w_1^Tx_1 + 1) \\\\
L_2 = &amp; \max(0, w_0^Tx_2 - w_2^Tx_2 + 1) + \max(0, w_1^Tx_2 - w_2^Tx_2 + 1) \\\\
L = &amp; (L_0 + L_1 + L_2)/3
\end{align}\]

<p>Since these examples are 1-dimensional, the data \(x_i\) and weights \(w_j\) are numbers. Looking at, for instance, \(w_0\), some terms above are linear functions of \(w_0\) and each is clamped at zero. We can visualize this as follows:</p>

<div class="fig figcenter fighighlight">
  <img src="/assets/svmbowl.png" />
  <div class="figcaption">
    1-dimensional illustration of the data loss. The x-axis is a single weight and the y-axis is the loss. The data loss is a sum of multiple terms, each of which is either independent of a particular weight, or a linear function of it that is thresholded at zero. The full SVM data loss is a 30,730-dimensional version of this shape.
  </div>
</div>

<p>As an aside, you may have guessed from its bowl-shaped appearance that the SVM cost function is an example of a <a href="http://en.wikipedia.org/wiki/Convex_function">convex function</a> There is a large amount of literature devoted to efficiently minimizing these types of functions, and you can also take a Stanford class on the topic ( <a href="http://stanford.edu/~boyd/cvxbook/">convex optimization</a> ). Once we extend our score functions \(f\) to Neural Networks our objective functions will become non-convex, and the visualizations above will not feature bowls but complex, bumpy terrains.</p>

<p><em>Non-differentiable loss functions</em>. As a technical note, you can also see that the <em>kinks</em> in the loss function (due to the max operation) technically make the loss function non-differentiable because at these kinks the gradient is not defined. However, the <a href="http://en.wikipedia.org/wiki/Subderivative">subgradient</a> still exists and is commonly used instead. In this class will use the terms <em>subgradient</em> and <em>gradient</em> interchangeably.</p>

<p><a name="optimization"></a></p>

<h3 id="optimization">Optimization</h3>

<p>To reiterate, the loss function lets us quantify the quality of any particular set of weights <strong>W</strong>. The goal of optimization is to find <strong>W</strong> that minimizes the loss function. We will now motivate and slowly develop an approach to optimizing the loss function. For those of you coming to this class with previous experience, this section might seem odd since the working example we’ll use (the SVM loss) is a convex problem, but keep in mind that our goal is to eventually optimize Neural Networks where we can’t easily use any of the tools developed in the Convex Optimization literature.</p>

<p><a name="opt1"></a></p>

<h4 id="strategy-1-a-first-very-bad-idea-solution-random-search">Strategy #1: A first very bad idea solution: Random search</h4>

<p>Since it is so simple to check how good a given set of parameters <strong>W</strong> is, the first (very bad) idea that may come to mind is to simply try out many different random weights and keep track of what works best. This procedure might look as follows:</p>

<div class="language-python highlighter-rouge"><div class="highlight"><pre class="highlight"><code><span class="c1"># assume X_train is the data where each column is an example (e.g. 3073 x 50,000)
# assume Y_train are the labels (e.g. 1D array of 50,000)
# assume the function L evaluates the loss function
</span>
<span class="n">bestloss</span> <span class="o">=</span> <span class="nb">float</span><span class="p">(</span><span class="s">"inf"</span><span class="p">)</span> <span class="c1"># Python assigns the highest possible float value
</span><span class="k">for</span> <span class="n">num</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1000</span><span class="p">):</span>
  <span class="n">W</span> <span class="o">=</span> <span class="n">np</span><span class="p">.</span><span class="n">random</span><span class="p">.</span><span class="n">randn</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">3073</span><span class="p">)</span> <span class="o">*</span> <span class="mf">0.0001</span> <span class="c1"># generate random parameters
</span>  <span class="n">loss</span> <span class="o">=</span> <span class="n">L</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">Y_train</span><span class="p">,</span> <span class="n">W</span><span class="p">)</span> <span class="c1"># get the loss over the entire training set
</span>  <span class="k">if</span> <span class="n">loss</span> <span class="o">&lt;</span> <span class="n">bestloss</span><span class="p">:</span> <span class="c1"># keep track of the best solution
</span>    <span class="n">bestloss</span> <span class="o">=</span> <span class="n">loss</span>
    <span class="n">bestW</span> <span class="o">=</span> <span class="n">W</span>
  <span class="k">print</span> <span class="s">'in attempt %d the loss was %f, best %f'</span> <span class="o">%</span> <span class="p">(</span><span class="n">num</span><span class="p">,</span> <span class="n">loss</span><span class="p">,</span> <span class="n">bestloss</span><span class="p">)</span>

<span class="c1"># prints:
# in attempt 0 the loss was 9.401632, best 9.401632
# in attempt 1 the loss was 8.959668, best 8.959668
# in attempt 2 the loss was 9.044034, best 8.959668
# in attempt 3 the loss was 9.278948, best 8.959668
# in attempt 4 the loss was 8.857370, best 8.857370
# in attempt 5 the loss was 8.943151, best 8.857370
# in attempt 6 the loss was 8.605604, best 8.605604
# ... (trunctated: continues for 1000 lines)
</span></code></pre></div></div>

<p>In the code above, we see that we tried out several random weight vectors <strong>W</strong>, and some of them work better than others. We can take the best weights <strong>W</strong> found by this search and try it out on the test set:</p>

<div class="language-python highlighter-rouge"><div class="highlight"><pre class="highlight"><code><span class="c1"># Assume X_test is [3073 x 10000], Y_test [10000 x 1]
</span><span class="n">scores</span> <span class="o">=</span> <span class="n">Wbest</span><span class="p">.</span><span class="n">dot</span><span class="p">(</span><span class="n">Xte_cols</span><span class="p">)</span> <span class="c1"># 10 x

