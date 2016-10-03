#load "network.fsx"

#load "persistence.fsx"
#load "mnist.fsx"

open MathNet.Numerics.LinearAlgebra
printfn "loading MNIST data..."
let data =  Mnist.getData()

let trd, vld, tsd = data

// #load "render.fsx"

// Render.renderImage (fst trd.[30])
// |> printfn "%d\n\n%s" (snd trd.[30]) 

let trainingData:Network.TrainingData = 
  trd
  |> Array.map (fun (d, l) -> (DenseMatrix.ofColumnArrays [| Array.map (fun px ->  (float px) ) d |], Network.toMatrix l) )


let testData:Network.TestData = 
  tsd
  |> Array.map (fun (d, l) -> (DenseMatrix.ofColumnArrays [| Array.map (fun px ->  (float px) ) d |], l) )

let net = Network.network [784;44;10]
printfn "Started training network"
let trainedNet = Network.SGD net trainingData 50 10 2.5 testData
printfn "Saving trained network"
Persistence.saveNet trainedNet "nets/trained-44N-50E.net"