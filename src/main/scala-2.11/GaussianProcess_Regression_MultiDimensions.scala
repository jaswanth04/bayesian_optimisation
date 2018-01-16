import breeze.linalg.{DenseMatrix, DenseVector, SliceMatrix, argmax, linspace, max}
import breeze.numerics.{pow, sin}
import breeze.plot.{Figure, plot}
import breeze.stats.distributions.{Gaussian, Rand}

object GaussianProcess_Regression_MultiDimensions {

  def main(args: Array[String]): Unit = {

    val x1 = linspace(-5,5,5)
    val x2 = linspace(-4,4,5)
    val x3 = linspace(-3,3,5)

    println(x1)
    println(x2)
    println(x3)

    val x = for (x <- x1.toArray; y <- x2.toArray; z <- x3.toArray) yield Array(x,y,z)

    val x_space: DenseMatrix[Double] = new DenseMatrix(5,25,x.flatten).t

    // Defining a sin function with some noise
    def f(x: DenseVector[Double]) = pow(x(0),5)*19 + sin(x(1)) * 0.9 * (x(1)-5) + pow(x(2),2)*4
    val noiseVariance = 0.0005
    def randGen = Rand.uniform
    def y(x: DenseVector[Double]) = f(x) + randGen.draw()*8000*noiseVariance

    // Generating training data as a subset of test data
    val x_train_indices = Seq(1,4,12,15,20)
    val x_train = x_space(x_train_indices,::).toDenseMatrix

    val y_train = new DenseVector(x_train_indices.map(x_space(_,::).t).toArray.map(y))

    // Instantiating new Gaussian Process function with Exponential Squared Kernel
    val GP = new GaussianProcess(ExpSquaredKernel(paramVector = DenseVector(0.8,0.8,0.8,0.8,0.8)),noiseVariance)

    // Generating Prior Samples
    val f_prior = GP.getPriorSample(x_space,4)

    println(f_prior)

    // Updating the parameters with the training Data
    GP.update(x_train,y_train)

    //Get mean, standard deviation and CoVariance for the test points
    val (mu,s,sigma) = GP.getMeanAndStandardDeviation(x_space)

    val x_test = linspace(-5,5,25)

    // Get posterior samples post training
    val f_posterior = GP.getPosteriorSample(x_space,4)

    println(f_posterior)

    // Training the Function on new Data Points
    val x_train_indices_2 = Seq(2,6,10,18,22)
    val x_train_2 = x_space(x_train_indices_2,::).toDenseMatrix
    println(x_train_2)
    val y_train_2: DenseVector[Double] = new DenseVector(x_train_indices_2.map(x_space(_,::).t).toArray.map(y))
    println(y_train_2)

    GP.update(x_train_2,y_train_2)

    val (mu_2,s_2,sigma_2) = GP.getMeanAndStandardDeviation(x_space)

    // Generating and plotting the posterior samples
    val f_posterior_2 = GP.getPosteriorSample(x_space,4)

    println(f_posterior_2)
  }


}
