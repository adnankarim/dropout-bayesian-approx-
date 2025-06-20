**Dropout as a Bayesian Approximation** shows that by keeping dropout turned on at test time and averaging many forward passes (“MC dropout”), a standard neural network not only makes predictions but also estimates its own uncertainty.

* **Key idea:** Training with dropout and weight decay matches doing a form of Bayesian inference.
* **At inference:** Run the same input through the network T times with different dropout masks, then compute the mean prediction and its variance.
* **Experiments replicated:**

  * Regression on 10 UCI datasets
  * Long-term CO₂ forecasting (Mauna Loa data)
  * Solar irradiance interpolation
  * MNIST digit-rotation uncertainty
  * A simple “Catch” reinforcement-learning task

**Main findings:**

* Predictive performance (RMSE, log-likelihood) matches earlier results.
* Uncertainty grows appropriately when you move away from the training data.
* Using unbounded activations (ReLU) gives rising uncertainty in unseen regions; bounded ones (Tanh) can under-estimate it.
* In RL, using uncertainty (Thompson sampling via MC dropout) explores more efficiently than a naïve ε-greedy strategy.

In short, MC dropout is a lightweight way to get both good predictions and calibrated uncertainty estimates from familiar networks.
