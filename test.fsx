#load "network.fsx"
#load "persistence.fsx"
#load "mnist.fsx"

open MathNet.Numerics.LinearAlgebra


let trainedNet = Persistence.loadNet "nets/trained-30N-30E.net"



let data =  Mnist.getData()

let trd, vld, tsd = data

#load "render.fsx"

// Render.renderImage (fst trd.[30])
// |> printfn "%d\n\n%s" (snd trd.[30]) 

let trainingData:Network.TrainingData = 
  trd
  |> Array.map (fun (d, l) -> (DenseMatrix.ofColumnArrays [| Array.map (fun px ->  (float px) ) d |], Network.toMatrix l) )




