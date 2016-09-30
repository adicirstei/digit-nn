let renderImage (img:byte []) : string =
  let grad = " .:oO"
  img
  |> Array.map (fun b -> string grad.[int ( (float b / 255.0) * 4.0) ])
  |> Array.chunkBySize 28
  |> Array.map (fun s -> String.concat "" s)
  |> String.concat "\n"