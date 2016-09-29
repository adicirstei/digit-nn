open System.IO
open System.IO.Compression

type MNISTImage = byte []

let toInt (ba: byte []) : int =
  ((int ba.[0]) <<< 24)
  ||| ((int ba.[1]) <<< 16)
  ||| ((int ba.[2]) <<< 8)
  ||| (int ba.[3])

let loadImageFile f : MNISTImage [] =
  let gz = new GZipStream(File.OpenRead(f), CompressionMode.Decompress)
  let ms = new MemoryStream()
  gz.CopyTo(ms)
  let arr = ms.ToArray()
  let magik = arr |> (Array.skip 0 >>  Array.take 4 )
  let len = arr |> (Array.skip 4 >>  Array.take 4 )
  let rows = arr |> (Array.skip 8 >>  Array.take 4 ) |> toInt
  let cols = arr |> (Array.skip 12 >>  Array.take 4 ) |> toInt

  arr
  |> Array.skip 16
  |>Array.chunkBySize (rows*cols)



let loadLabelFile f : byte [] =
  let gz = new GZipStream(File.OpenRead(f), CompressionMode.Decompress)
  let ms = new MemoryStream()
  gz.CopyTo(ms)
  let arr = ms.ToArray()
  let magik = Array.take 4 arr
  let len = arr |> (Array.skip 4 >>  Array.take 4 )

  arr
  |> Array.skip 8


let getData () =
  let trLbls = loadLabelFile "data/train-labels-idx1-ubyte.gz"
  let tsLbls = loadLabelFile "data/t10k-labels-idx1-ubyte.gz"
  let trImag = loadImageFile "data/train-images-idx3-ubyte.gz"
  let tsImag = loadImageFile "data/t10k-images-idx3-ubyte.gz"
  let trDataFull = Array.zip trImag trLbls
  let tsData = Array.zip tsImag tsLbls
  (Array.take 50000 trDataFull, Array.skip 50000 trDataFull, tsData)