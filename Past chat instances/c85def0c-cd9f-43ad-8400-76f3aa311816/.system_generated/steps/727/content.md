Title: Live Content

Description: Fetched live

Source: https://cs231n.github.io/neural-networks-3/

---

<!DOCTYPE html>
<html>

  <head>
  <meta charset="utf-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <title>CS231n Deep Learning for Computer Vision</title>
  <meta name="viewport" content="width=device-width">
  <meta name="description" content="Course materials and notes for Stanford class CS231n: Deep Learning for Computer Vision.">
  <link rel="canonical" href="https://cs231n.github.io/neural-networks-3/">

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
  <li><a href="#gradcheck">Gradient checks</a></li>
  <li><a href="#sanitycheck">Sanity checks</a></li>
  <li><a href="#baby">Babysitting the learning process</a>
    <ul>
      <li><a href="#loss">Loss function</a></li>
      <li><a href="#accuracy">Train/val accuracy</a></li>
      <li><a href="#ratio">Weights:Updates ratio</a></li>
      <li><a href="#distr">Activation/Gradient distributions per layer</a></li>
      <li><a href="#vis">Visualization</a></li>
    </ul>
  </li>
  <li><a href="#update">Parameter updates</a>
    <ul>
      <li><a href="#sgd">First-order (SGD), momentum, Nesterov momentum</a></li>
      <li><a href="#anneal">Annealing the learning rate</a></li>
      <li><a href="#second">Second-order methods</a></li>
      <li><a href="#ada">Per-parameter adaptive learning rates (Adagrad, RMSProp)</a></li>
    </ul>
  </li>
  <li><a href="#hyper">Hyperparameter Optimization</a></li>
  <li><a href="#eval">Evaluation</a>
    <ul>
      <li><a href="#ensemble">Model Ensembles</a></li>
    </ul>
  </li>
  <li><a href="#summary">Summary</a></li>
  <li><a href="#add">Additional References</a></li>
</ul>

<h2 id="learning">Learning</h2>

<p>In the previous sections we’ve discussed the static parts of a Neural Networks: how we can set up the network connectivity, the data, and the loss function. This section is devoted to the dynamics, or in other words, the process of learning the parameters and finding good hyperparameters.</p>

<p><a name="gradcheck"></a></p>

<h3 id="gradient-checks">Gradient Checks</h3>

<p>In theory, performing a gradient check is as simple as comparing the analytic gradient to the numerical gradient. In practice, the process is much more involved and error prone. Here are some tips, tricks, and issues to watch out for:</p>

<p><strong>Use the centered formula</strong>. The formula you may have seen for the finite difference approximation when evaluating the numerical gradient looks as follows:</p>

\[\frac{df(x)}{dx} = \frac{f(x + h) - f(x)}{h} \hspace{0.1in} \text{(bad, do not use)}\]

<p>where \(h\) is a very small number, in practice approximately 1e-5 or so. In practice, it turns out that it is much better to use the <em>centered</em> difference formula of the form:</p>

\[\frac{df(x)}{dx} = \frac{f(x + h) - f(x - h)}{2h} \hspace{0.1in} \text{(use instead)}\]

<p>This requires you to evaluate the loss function twice to check every single dimension of the gradient (so it is about 2 times as expensive), but the gradient approximation turns out to be much more precise. To see this, you can use Taylor expansion of \(f(x+h)\) and \(f(x-h)\) and verify that the first formula has an error on order of \(O(h)\), while the second formula only has error terms on order of \(O(h^2)\) (i.e. it is a second order approximation).</p>

<p><strong>Use relative error for the comparison</strong>. What are the details of comparing the numerical gradient \(f’_n\) and analytic gradient \(f’_a\)? That is, how do we know if the two are not compatible? You might be temped to keep track of the difference \(\mid f’_a - f’_n \mid \) or its square and define the gradient check as failed if that difference is above a threshold. However, this is problematic. For example, consider the case where their difference is 1e-4. This seems like a very appropriate difference if the two gradients are about 1.0, so we’d consider the two gradients to match. But if the gradients were both on order of 1e-5 or lower, then we’d consider 1e-4 to be a huge difference and likely a failure. Hence, it is always more appropriate to consider the <em>relative error</em>:</p>

\[\frac{\mid f'_a - f'_n \mid}{\max(\mid f'_a \mid, \mid f'_n \mid)}\]

<p>which considers their ratio of the differences to the ratio of the absolute values of both gradients. Notice that normally the relative error formula only includes one of the two terms (either one), but I prefer to max (or add) both to make it symmetric and to prevent dividing by zero in the case where one of the two is zero (which can often happen, especially with ReLUs). However, one must explicitly keep track of the case where both are zero and pass the gradient check in that edge case. In practice:</p>

<ul>
  <li>relative error &gt; 1e-2 usually means the gradient is probably wrong</li>
  <li>1e-2 &gt; relative error &gt; 1e-4 should make you feel uncomfortable</li>
  <li>1e-4 &gt; relative error is usually okay for objectives with kinks. But if there are no kinks (e.g. use of tanh nonlinearities and softmax), then 1e-4 is too high.</li>
  <li>1e-7 and less you should be happy.</li>
</ul>

<p>Also keep in mind that the deeper the network, the higher the relative errors will be. So if you are gradient checking the input data for a 10-layer network, a relative error of 1e-2 might be okay because the errors build up on the way. Conversely, an error of 1e-2 for a single differentiable function likely indicates incorrect gradient.</p>

<p><strong>Use double precision</strong>. A common pitfall is using single precision floating point to compute gradient check. It is often that case that you might get high relative errors (as high as 1e-2) even with a correct gradient implementation. In my experience I’ve sometimes seen my relative errors plummet from 1e-2 to 1e-8 by switching to double precision.</p>

<p><strong>Stick around active range of floating point</strong>. It’s a good idea to read through <a href="http://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html">“What Every Computer Scientist Should Know About Floating-Point Arithmetic”</a>, as it may demystify your errors and enable you to write more careful code. For example, in neural nets it can be common to normalize the loss function over the batch. However, if your gradients per datapoint are very small, then <em>additionally</em> dividing them by the number of data points is starting to give very small numbers, which in turn will lead to more numerical issues. This is why I like to always print the raw numerical/analytic gradient, and make sure that the numbers you are comparing are not extremely small (e.g. roughly 1e-10 and smaller in absolute value is worrying). If they are you may want to temporarily scale your loss function up by a constant to bring them to a “nicer” range where floats are more dense - ideally on the order of 1.0, where your float exponent is 0.</p>

<p><strong>Kinks in the objective</strong>. One source of inaccuracy to be aware of during gradient checking is the problem of <em>kinks</em>. Kinks refer to non-differentiable parts of an objective function, introduced by functions such as ReLU (\(max(0,x)\)), or the SVM loss, Maxout neurons, etc. Consider gradient checking the ReLU function at \(x = -1e6\). Since \(x &lt; 0\), the analytic gradient at this point is exactly zero. However, the numerical gradient would suddenly compute a non-zero gradient because \(f(x+h)\) might cross over the kink (e.g. if \(h &gt; 1e-6\)) and introduce a non-zero contribution. You might think that this is a pathological case, but in fact this case can be very common. For example, an SVM for CIFAR-10 contains up to 450,000 \(max(0,x)\) terms because there are 50,000 examples and each example yields 9 terms to the objective. Moreover, a Neural Network with an SVM classifier will contain many more kinks due to ReLUs.</p>

<p>Note that it is possible to know if a kink was crossed in the evaluation of the loss. This can be done by keeping track of the identities of all “winners” in a function of form \(max(x,y)\); That is, was x or y higher during the forward pass. If the identity of at least one winner changes when evaluating \(f(x+h)\) and then \(f(x-h)\), then a kink was crossed and the numerical gradient will not be exact.</p>

<p><strong>Use only few datapoints</strong>. One fix to the above problem of kinks is to use fewer datapoints, since loss functions that contain kinks (e.g. due to use of ReLUs or margin losses etc.) will have fewer kinks with fewer datapoints, so it is less likely for you to cross one when you perform the finite different approximation. Moreover, if your gradcheck for only ~2 or 3 datapoints then you would almost certainly gradcheck for an entire batch. Using very few datapoints also makes your gradient check faster and more efficient.</p>

<p><strong>Be careful with the step size h</strong>. It is not necessarily the case that smaller is better, because when \(h\) is much smaller, you may start running into numerical precision problems. Sometimes when the gradient doesn’t check, it is possible that you change \(h\) to be 1e-4 or 1e-6 and suddenly the gradient will be correct. This <a href="http://en.wikipedia.org/wiki/Numerical_differentiation">wikipedia article</a> contains a chart that plots the value of <strong>h</strong> on the x-axis and the numerical gradient error on the y-axis.</p>

<p><strong>Gradcheck during a “characteristic” mode of operation</strong>. It is important to realize that a gradient check is performed at a particular (and usually random), single point in the space of parameters. Even if the gradient check succeeds at that point, it is not immediately certain that the gradient is correctly implemented globally. Additionally, a random initialization might not be the most “characteristic” point in the space of parameters and may in fact introduce pathological situations where the gradient seems to be correctly implemented but isn’t. For instance, an SVM with very small weight initialization will assign almost exactly zero scores to all datapoints and the gradients will exhibit a particular pattern across all datapoints. An incorrect implementation of the gradient could still produce this pattern and not generalize to a more characteristic mode of operation where some scores are larger than others. Therefore, to be safe it is best to use a short <strong>burn-in</strong> time during which the network is allowed to learn and perform the gradient check after the loss starts to go down. The danger of performing it at the first iteration is that this could introduce pathological edge cases and mask an incorrect implementation of the gradient.</p>

<p><strong>Don’t let the regularization overwhelm the data</strong>. It is often the case that a loss function is a sum of the data loss and the regularization loss (e.g. L2 penalty on weights). One danger to be aware of is that the regularization loss may overwhelm the data loss, in which case the gradients will be primarily coming from the regularization term (which usually has a much simpler gradient expression). This can mask an incorrect implementation of the data loss gradient. Therefore, it is recommended to turn off regularization and check the data loss alone first, and then the regularization term second and independently. One way to perform the latter is to hack the code to remove the data loss contribution. Another way is to increase the regularization strength so as to ensure that its effect is non-negligible in the gradient check, and that an incorrect implementation would be spotted.</p>

<p><strong>Remember to turn off dropout/augmentations</strong>. When performing gradient check, remember to turn off any non-deterministic effects in the network, such as dropout, random data augmentations, etc. Otherwise these can clearly introduce huge errors when estimating the numerical gradient. The downside of turning off these effects is that you wouldn’t be gradient checking them (e.g. it might be that dropout isn’t backpropagated correctly). Therefore, a better solution might be to force a particular random seed before evaluating both \(f(x+h)\) and \(f(x-h)\), and when evaluating the analytic gradient.</p>

<p><strong>Check only few dimensions</strong>. In practice the gradients can have sizes of million parameters. In these cases it is only practical to check some of the dimensions of the gradient and assume that the others are correct. <strong>Be careful</strong>: One issue to be careful with is to make sure to gradient check a few dimensions for every separate parameter. In some applications, people combine the parameters into a single large parameter vector for convenience. In these cases, for example, the biases could only take up a tiny number of parameters from the whole vector, so it is important to not sample at random but to take this into account and check that all parameters receive the correct gradients.</p>

<p><a name="sanitycheck"></a></p>

<h3 id="before-learning-sanity-checks-tipstricks">Before learning: sanity checks Tips/Tricks</h3>

<p>Here are a few sanity checks you might consider running before you plunge into expensive optimization:</p>

<ul>
  <li><strong>Look for correct loss at chance performance.</strong> Make sure you’re getting the loss you expect when you initialize with small parameters. It’s best to first check the data loss alone (so set regularization strength to zero). For example, for CIFAR-10 with a Softmax classifier we would expect the initial loss to be 2.302, because we expect a diffuse probability of 0.1 for each class (since there are 10 classes), and Softmax loss is the negative log probability of the correct class so: -ln(0.1) = 2.302. For The Weston Watkins SVM, we expect all desired margins to be violated (since all scores are approximately zero), and hence expect a loss of 9 (since margin is 1 for each wrong class). If you’re not seeing these losses there might be issue with initialization.</li>
  <li>As a second sanity check, increasing the regularization strength should increase the loss</li>
  <li><strong>Overfit a tiny subset of data</strong>. Lastly and most importantly, before training on the full dataset try to train on a tiny portion (e.g. 20 examples) of your data and make sure you can achieve zero cost. For this experiment it’s also best to set regularization to zero, otherwise this can prevent you from getting zero cost. Unless you pass this sanity check with a small dataset it is not worth proceeding to the full dataset. Note that it may happen that you can overfit very small dataset but still have an incorrect implementation. For instance, if your datapoints’ features are random due to some bug, then it will be possible to overfit your small training set but you will never notice any generalization when you fold it your full dataset.</li>
</ul>

<p><a name="baby"></a></p>

<h3 id="babysitting-the-learning-process">Babysitting the learning process</h3>

<p>There are multiple useful quantities you should monitor during training of a neural network. These plots are the window into the training process and should be utilized to get intuitions about different hyperparameter settings and how they should be changed for more efficient learning.</p>

<p>The x-axis of the plots below are always in units of epochs, which measure how many times every example has been seen during training in expectation (e.g. one epoch means that every example has been seen once). It is preferable to track epochs rather than iterations since the number of iterations depends on the arbitrary setting of batch size.</p>

<p><a name="loss"></a></p>

<h4 id="loss-function">Loss function</h4>

<p>The first quantity that is useful to track during training is the loss, as it is evaluated on the individual batches during the forward pass. Below is a cartoon diagram showing the loss over time, and especially what the shape might tell you about the learning rate:</p>

<div class="fig figcenter fighighlight">
  <img src="/assets/nn3/learningrates.jpeg" width="49%" />
  <img src="/assets/nn3/loss.jpeg" width="49%" />
  <div class="figcaption">
    <b>Left:</b> A cartoon depicting the effects of different learning rates. With low learning rates the improvements will be linear. With high learning rates they will start to look more exponential. Higher learning rates will decay the loss faster, but they get stuck at worse values of loss (green line). This is because there is too much "energy" in the optimization and the parameters are bouncing around chaotically, unable to settle in a nice spot in the optimization landscape. <b>Right:</b> An example of a typical loss function over time, while training a small network on CIFAR-10 dataset. This loss function looks reasonable (it might indicate a slightly too small learning rate based on its speed of decay, but it's hard to say), and also indicates that the batch size might be a little too low (since the cost is a little too noisy).
  </div>
</div>

<p>The amount of “wiggle” in the loss is related to the batch size. When the batch size is 1, the wiggle will be relatively high. When the batch size is the full dataset, the wiggle will be minimal because every gradient update should be improving the loss function monotonically (unless the learning rate is set too high).</p>

<p>Some people prefer to plot their loss functions in the log domain. Since learning progress generally takes an exponential form shape, the plot appears as a slightly more interpretable straight line, rather than a hockey stick. Additionally, if multiple cross-validated models are plotted on the same loss graph, the differences between them become more apparent.</p>

<p>Sometimes loss functions can look funny <a href="http://lossfunctions.tumblr.com/">lossfunctions.tumblr.com</a>.</p>

<p><a name="accuracy"></a></p>

<h4 id="trainval-accuracy">Train/Val accuracy</h4>

<p>The second important quantity to track while training a classifier is the validation/training accuracy. This plot can give you valuable insights into the amount of overfitting in your model:</p>

<div class="fig figleft fighighlight">
  <img src="/assets/nn3/accuracies.jpeg" />
  <div class="figcaption">
    The gap between the training and validation accuracy indicates the amount of overfitting. Two possible cases are shown in the diagram on the left. The blue validation error curve shows very small validation accuracy compared to th

