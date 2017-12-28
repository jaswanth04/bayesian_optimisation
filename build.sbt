name := "bayesian_optimisation"

version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq("org.scalatest" %% "scalatest" % "3.0.1" % "test",
  "org.scalanlp" % "breeze_2.11" % "0.12",
  "org.scalanlp" % "breeze-natives_2.11" % "0.12",
  "org.slf4j" % "slf4j-api" % "1.7.22",
  "org.slf4j" % "slf4j-simple" % "1.7.22",
  "org.scalanlp" %% "breeze-viz" % "0.12")

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
