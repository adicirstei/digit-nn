#I "packages/FsPickler/lib/net45"
#r "FsPickler.dll"

#load "network.fsx"

open MBrace.FsPickler
let saveNet net file =
  let stream = new System.IO.FileStream(file, System.IO.FileMode.OpenOrCreate)
  let binary = FsPickler.CreateBinarySerializer()
  binary.Serialize(stream, net)
  net


let loadNet file =
  let stream = new System.IO.FileStream(file, System.IO.FileMode.Open)
  let binary = FsPickler.CreateBinarySerializer()
  binary.Deserialize<Network.Net>(stream)
