import breeze.linalg.{Axis, DenseMatrix, diag, mpow, sum}
import breeze.numerics.{exp,pow}

object trial {

  def main(args: Array[String]): Unit = {

    val a = DenseMatrix((1,2),(3,4))
//    val b = DenseMatrix((4,5),(6,7))
//
//    println(a.t)
//    println("---------")
//    println(b.t)
//    println("----------")
//    println((a*b).t)
//    println("-------------")
//    println(b.t * a.t)
//
//    val c: DenseMatrix[Double] = a.mapValues(math.pow(_,2))

//    println(exp(a))

    println(pow(a,2))


  }

}
