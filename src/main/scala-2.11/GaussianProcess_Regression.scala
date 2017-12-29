import breeze.linalg.{DenseVector,linspace}
import breeze.numerics.sin
import breeze.plot.{Figure, plot}
import breeze.stats.distributions.Rand


object GaussianProcess_Regression {

  def main(args: Array[String]): Unit = {

    // Generating points between -5,5
    val n = 50
    val x_test = linspace(-5,5,n)

    // Defining a sin function with some noise
    def f(x: Double) = sin(x) * 0.9 * (x-5)
    val noiseVariance = 0.0005
    def randGen = Rand.uniform
    def y(x: Double) = f(x) + randGen.draw()*8000*noiseVariance

    // Generating training data as a subset of test data
    val x_train_indices = Array(4,18,29,35,47)
    val x_train = new DenseVector(x_train_indices.map(x_test(_)))
    val y_train: DenseVector[Double] = x_train.map(y)

    // Instantiating new Gaussian Process function with Exponential Squared Kernel
    val GP = new GaussianProcess(ExpSquaredKernel(0.1),noiseVariance)

    // Generating Prior Samples
    val f_prior = GP.getPriorSample(x_test,4)

    // Updating the parameters with the training Data
    GP.update(x_train,y_train)

    //Get mean, standard deviation and CoVariance for the test points
    val (mu,s,sigma) = GP.getMeanAndStandardDeviation(x_test)

    // Get posterior samples post training
    val f_posterior = GP.getPosteriorSample(x_test,4)

    //Plotting the data
    val fig = Figure()

    //Plotting the true function
    val plt3 = fig.subplot(2,2,0)
    plt3 += plot(x_test,x_test.map(y), name = "true Function", style = '-')
    plt3.title = "True Function"

    // Plotting the Priors
    val plt2 = fig.subplot(2,2,1)

    for (prior <- f_prior)
      plt2 += plot(x_test,prior)

    plt2.title = "Prior"

    // Plotting the mean and boundaries
    val plt = fig.subplot(2,2,2)

    plt += plot(x_test,mu,name = "mean")
    plt += plot(x_test,mu + s * 2.0, name = "upper boundary",style = '-')
    plt += plot(x_test,mu - s * 2.0, name = "upper boundary",style = '-')

    plt.title = "Mean and boundaries - After first training"
    plt.legend = true

    //Plotting the Posterior
    val plt1 = fig.subplot(2,2,3)

    for (posterior <- f_posterior)
      plt1 += plot(x_test,posterior)

    plt1.title = "Posterior - After first training"

    fig.refresh()

    // Training the Function on new Data Points
    val x_train_indices_2 = Array(25,22,17,19,10)
    val x_train_2 = new DenseVector(x_train_indices_2.map(x_test(_)))
    val y_train_2: DenseVector[Double] = x_train_2.map(y)

    GP.update(x_train_2,y_train_2)

    val (mu_2,s_2,sigma_2) = GP.getMeanAndStandardDeviation(x_test)

    // Plotting the Mean and Boundaries after the second training
    val plt4 = fig.subplot(3,2,4)

    plt4 += plot(x_test,mu_2,name = "mean")
    plt4 += plot(x_test,mu_2 + s_2 * 2.0, name = "upper boundary",style = '-')
    plt4 += plot(x_test,mu_2 - s_2 * 2.0, name = "upper boundary",style = '-')

    plt4.title = "Mean and boundaries - After Second training"
    plt4.legend = true

    // Generating and plotting the posterior samples
    val f_posterior_2 = GP.getPosteriorSample(x_test,4)

    val plt5 = fig.subplot(3,2,5)

    for (posterior <- f_posterior_2)
      plt5 += plot(x_test,posterior)

    plt5.title = "Posterior - After Second training"

    fig.refresh()

  }
}
