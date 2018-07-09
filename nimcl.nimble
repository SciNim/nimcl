version       = "0.1.2"
author        = "Andrea Ferretti"
description   = "OpenCL utilities"
license       = "Apache2"
skipDirs      = @["examples"]
skipFiles     = @["points.json"]

requires "nim >= 0.13.0", "opencl >= 1.0"


template dependsOn*(task: untyped): typed =
  exec "nimble " & astToStr(task)

proc addDefaults() =
  switch("cincludes", "/usr/local/cuda/targets/x86_64-linux/include")
  switch("clibdir", "/usr/local/cuda/targets/x86_64-linux/lib")
  --define: release
  --path: "."

task info, "OpenCL info":
  addDefaults()
  --run
  setCommand "c", "examples/info.nim"

task clcompile, "OpenCL compiler":
  addDefaults()
  setCommand "c", "examples/compile.nim"

task vadd, "run vector add example":
  addDefaults()
  --run
  setCommand "c", "examples/vadd.nim"

task headers, "compile headers with c2nim":
  exec "c2nim examples/point.h"

task kmeans, "run kmeans example":
  dependsOn headers
  addDefaults()
  --run
  setCommand "c", "examples/kmeans.nim"