# Copyright 2016 UniCredit S.p.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, math, nimcl

proc main() =
  const
    body = staticRead("vadd.cl")
    size = 1_000_000
  var
    a = newSeq[float32](size)
    b = newSeq[float32](size)
    c = newSeq[float32](size)

  for i in 0 .. a.high:
    a[i] = i.float32
    b[i] = (i * i).float32

  let
    (device, context, queue) = singleDeviceDefaults()
    program = context.createAndBuild(body, device)
    add = program.createKernel("add_vector")
    gpuA = context.bufferLike(a)
    gpuB = context.bufferLike(b)
    gpuC = context.bufferLike(c)

  add.args(gpuA, gpuB, gpuC, size.int32)

  queue.write(a, gpuA)
  queue.write(b, gpuB)
  queue.run(add, size)
  queue.read(c, gpuC)

  echo c[1 .. 100]

  # Clean up
  release(queue)
  release(add)
  release(program)
  release(gpuA)
  release(gpuB)
  release(gpuC)
  release(context)

when isMainModule:
  main()