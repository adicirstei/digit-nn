#load "packages/MathNet.Numerics.FSharp/MathNet.Numerics.fsx"

open MathNet.Numerics.LinearAlgebra
//open MathNet.Numerics.LinearAlgebra.Double

type Net = {
  numLayers : int
  sizes : int list
  biases : (Matrix<float>) list
  weights : (Matrix<float>) list
}

let network sizes =
  {
    numLayers = List.length sizes
    sizes = sizes
    biases = [for y in sizes.[1..] -> DenseMatrix.randomStandard<float>y 1]
    weights = [for x, y in List.zip sizes.[1..] (List.take (sizes.Length - 1) sizes)  -> 
                DenseMatrix.randomStandard<float> y x ]
  }


let sigmoidE (z:float) : float = 1.0 / (1.0 + exp (-z))
let sigmoid : (Matrix<float> -> Matrix<float>) = Matrix.map sigmoidE

let feedforward net a =
  List.zip net.biases net.weights 
  |> List.fold (fun s (b,w) -> sigmoid ((w * s) + b) ) a

let net = network [2;3;1]

feedforward net (DenseMatrix.randomStandard<float> 1 2)
