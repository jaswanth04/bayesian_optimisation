import breeze.numerics.{exp, pow}

case class ExpSquaredKernel(param: Double = 0.1) extends Kernel {

  def _kernel(x: Double, y: Double): Double = exp(-0.5*(1/param)*pow(x-y,2))

}
