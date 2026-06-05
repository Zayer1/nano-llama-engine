Title: Live Content

Description: Fetched live

Source: https://cs231n.github.io/

---

<!DOCTYPE html>
<html>

  <head>
  <meta charset="utf-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <title>CS231n Deep Learning for Computer Vision</title>
  <meta name="viewport" content="width=device-width">
  <meta name="description" content="Course materials and notes for Stanford class CS231n: Deep Learning for Computer Vision.">
  <link rel="canonical" href="https://cs231n.github.io/">

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
      <div>

  These notes accompany the Stanford CS class <a href="http://cs231n.stanford.edu/">CS231n: Deep Learning for Computer Vision</a>. For questions/concerns/bug reports, please submit a pull request directly to
  our <a href="https://github.com/cs231n/cs231n.github.io">git repo</a>.
  <br>
  <!-- For questions/concerns/bug reports contact <a href="http://cs.stanford.edu/people/jcjohns/">Justin Johnson</a> regarding the assignments, or contact <a href="http://cs.stanford.edu/people/karpathy/">Andrej Karpathy</a> regarding the course notes. You can also submit a pull request directly to our <a href="https://github.com/cs231n/cs231n.github.io">git repo</a>. -->
  <!-- <br> -->
  <!-- We encourage the use of the <a href="https://hypothes.is/">hypothes.is</a> extension to annote comments and discuss these notes inline. -->
</div>

<div class="home">
  <div class="materials-wrap">
    <div class="module-header">Spring 2026 Assignments</div>
    <div class="materials-item">
      <a href="assignments2026/assignment1/">Assignment #1: Image Classification, kNN, Softmax, Fully-Connected Neural Network, Fully-Connected Nets</a>
    </div>
    <div class="materials-item">
      <a href="assignments2026/assignment2/">Assignment #2: Batch Normalization, Dropout, Convolutional Nets, Network Visualization, Image Captioning with RNNs</a>
    </div>
    <div class="materials-item">
      <a href="#">Image Captioning with Transformers, Self-Supervised Learning, Diffusion Models, CLIP and DINO Models (Releasing May 14)</a>
    </div>
  </div>

  <!-- <div class="materials-wrap">
    <div class="module-header">Spring 2021 Assignments</div>
      <div class="materials-item">
        <a href="assignments2021/assignment1/">Assignment #1: Image Classification, kNN, SVM, Softmax, Fully Connected Neural Network</a>
      </div>
      <div class="materials-item">
        <a href="assignments2021/assignment2/">Assignment #2: Fully Connected and Convolutional Nets, Batch Normalization, Dropout, Frameworks</a>
      </div>
      <div class="materials-item">
        <a href="assignments2021/assignment3/">Assignment #3: Image Captioning with RNNs and Transformers, Network Visualization,
          Generative Adversarial Networks, Self-Supervised Contrastive Learning</a>
      </div>
  </div> -->
  <!--
    <div class="materials-item">
      <a href="assignments2019/assignment2/">
        Assignment #2: 

