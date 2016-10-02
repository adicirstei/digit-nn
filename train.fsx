#load "network.fsx"

#load "persistence.fsx"
#load "mnist.fsx"

open MathNet.Numerics.LinearAlgebra

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

let net = Network.network [784;100;10]

let trainedNet = Network.SGD net trainingData 50 10 5.0 testData

Persistence.saveNet trainedNet "nets/trained-100N-50E.net"