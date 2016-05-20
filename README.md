# Nim OpenCL utilities

This is an attempt at a high level wrapper over
[OpenCL](https://github.com/nim-lang/opencl/).

For now, things are added when needed and as such, they may not be perfectly
coherent. Still, they should be enough to cover the simplest cases and get
started.

Some API changes can also be expected as the library becomes more
comprehensive.

## Vector add example

The "hello, world!" of OpenCL:

```nim
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
```