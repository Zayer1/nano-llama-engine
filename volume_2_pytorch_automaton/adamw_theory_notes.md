# AdamW Calculus: Theoretical Mechanics

*Write your conceptual understanding of the four steps of AdamW below. Use plain English, logic, or metaphors. Don't worry about Python syntax.*

In order to understand adamw optimizer, first we must get to its predecessor.
Traditionally, in order to penalize weights to prevent overfitting, we use a method called standard L2 regularization. Let's call it regularized Loss: L_reg
L_reg(theta) = L(theta) + lambda/2 * theta.T * theta
Here, lambda is the weight decay coefficient
Now we apply backprops to it with respect to theta, the function will be
nabla_dL_reg = nabla_L(theta) + lambda * theta (Note: Nabla is like a 1D Jacobian matrix, it's the multi-dimensional gradient, eta is the learning rate, and theta is the parameter)
Supposed the backprop to it is at step t, the function would be
g_t = Nabla_L(theta_t) + lamda * theta_t
If we were using Stochastic Gradient Descent, it'd be
g_t+1 = Theta_t - eta * dL/dL_reg_t = theta_t - eta * (nabla_L(theta_t) + lambda * theta_t)

Next, we should also know about bias correction
Adam maintains an Exponential Moving Average (EMA) of squared gradients, which we call the second moment or uncertained variance(v_t)
At first moment, let's set it at 0: v_0 = 0
The formula to update it to step 1 is: v_t = beta_2 * v_t-1 + (1 - beta_2) * gradient_t ** 2
In standard Adam, the decay rate beta_2 is usually set to 0.999. Now let's look at what happens on the very first training step (t=1)
v1 = 0.999*(0) + (1-0.999) * gradient_1 ** 2 = 0.001 * g_1**2
Because we started at 0, the tracker is biased and only captures 1/1000th of the true gradient variance
To fix this, Adam computes a corrected variance vt^ by dividing out the bias factor
vt^ = v1/(1-beta_2**t)
At step 1, this looks like vt^ = v1/(1-0.999**1) = v1/0.001, which scales back up nicely.

Now let's get back up to Adam
Adam (Adaptive Moment Estimation) combines the directional momentum of past gradients with a variance tracker to adapt the learning rate for every single parameter
Before we begin, let's initialize our movement tracker as arrays of zeros
m0 = 0 (First moment/momentum)
v0 = 0 (Second movement/Uncentered Variance)

Now we'll adapt 5-step update:
Step 1: Get the raw gradient
g_t = nabla_L(theta_t)
Step 2: Update the first moment
m_t = beta_1 * m_t-1 + (1-beta_1) * g_t
Standard beta_1 is usually 0.9
Step 3: Update the second moment
v_t = beta_2 * v_t-1 + (1-beta_2) * g_t**2
Standard beta_2 is usually 0.999
Step 4: Apply bias correction
m_t^ = m_t/(1-beta_1**t)
v_t^ = v_t/(1-beta_2**t)
Step 5: The final Adam update rule
We update the parameters by stepping in the direction of the momentum (m_t^), scaled down by volatility (sqrt(v_t^))
theta_t+1 = theta_t - eta * (m_t^/sqrt(v_t^)+1e-9)

If we try to use standard L2 regularization inside Adam, our gradient becomes g_t = nabla_L(theta_t) + lambda * theta_t
If we plug that g_t into Adam's step 3, the weight decay penalty (lambda*theta_t) gets squared inside v_t, and then shoved into the denominator of step 5 (sqrt(v_t^)). This mathematically ruins the regularization because weights with massive gradients get a massive denominator, causing their weight decay to shrink to almost nothing.

To fix this, AdamW skips putting the decay in step 1, uses the pure gradient for the moving averages, and manually subtracts the weight decay at the very end of step 5:
theta_t+1 = theta_t - eta * (m_t^/(sqrt(v_t^)+1e-9)) - eta * lambda * theta_t

Let's do a live example
Supposed we have 2 parameters in our model
theta_noisy: a weight that is experiencing massive and chaotic gradients from the loss function
theta_quiet: a weight that has practically 0 gradient from the loss function because it has already found a good local minimum

In standard Adam with L2 regularization, the weight decay penalty is added directly into the raw gradient at step 1:
g_t = nabla_L(theta_t) + lambda*theta_t
Now let's look at what happens to our 2 weights when they hit Adam's variance tracker ( step 3) and the final update fraction (step 5)
For theta_noisy: Because the loss function is massive, its variance tracker explodes into a huge number. When we calculate the final step size, the entire gradient - including the weight decay penalty - gets divided by sqrt(v_t^). Because we are dividing by a massive number, the weight decay shrinks to almost nothing, which means the noisy weight escapes the penalty
For theta_quiet: Because the loss function is tiny, its variance tracker (v_t^) stays very small, and then when we calculate the final step, the gradient (which is mostly just lambda*theta_t at this point) is divided by a tiny number. Which causes the gradient to violently shoot up, and the quiet weight is penalized to almost 0.
This is the opposite of what we want to do.

Thus, a fix was proposed. Ilya Loshchilov and Frank Hutter realized that the variance checker should only adapt the learning rate based on the actual loss landscape, not the artificial weight decay penalty. To fix this, AdamW surgically removes the weight decay from the gradient calculation

Step A: the pure gradients
Instead of mixing in the penalty, AdamW calculates the pure gradient of the loss:
g_t = nabla_L(theta_t)
Step B: the pure moments
Because g_t is pure, the momentum (m_t^) and the variance checker (v_t^) are calculated strictly based on the performance of the model, they're no longer polluted by the artificial penalty
Step C: the decoupled update
AdamW calculates the adaptive step exactly like Adam, but it added a completely separated, independent math operation at the very end to apply the weight decay
theta_t+1 = theta_t - eta * (m_t^/(sqrt(v_t^)+1e-9)) - eta * lambda * theta_t
Look closely at - eta * lambda * theta_t
There is no division by sqrt(v_t^). By extracting the weight decay penalty and placing it directly outside the adaptive fraction, AdamW guarantees that every single weight in the entire network decays at the exact same rate (eta * lambda), regardless of whether the gradient is massive or not.
So, theta_noisy loses exactly eta * lambda * theta_noisy, and theta_quiet loses exactly eta * lambda * theta_quiet.