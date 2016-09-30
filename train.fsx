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

let net = Network.network [28*28;30;10]

let trainedNet = Network.SGD net trainingData 30 10 3.0 

Persistence.saveNet trainedNet "nets/trained-30N-30E.net"