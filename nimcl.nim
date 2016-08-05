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

import opencl

type
  PlatformNotFound = object of Exception
  DeviceNotFound = object of Exception

proc newPlatformNotFound(): ref PlatformNotFound =
  new result
  result.msg = "PlatformNotFound"

proc newDeviceNotFound(): ref DeviceNotFound =
  new result
  result.msg = "DeviceNotFound"

proc name*(id: PPlatformId): string =
  var size = 0
  check getPlatformInfo(id, PLATFORM_NAME, 0, nil, addr size)
  result = newString(size)
  check getPlatformInfo(id, PLATFORM_NAME, size, addr result[0], nil)

proc name*(id: PDeviceId): string =
  var size = 0
  check getDeviceInfo(id, DEVICE_NAME, 0, nil, addr size)
  result = newString(size)
  check getDeviceInfo(id, DEVICE_NAME, size, addr result[0], nil)

proc maxWorkGroups*(id: PDeviceId): int =
  check getDeviceInfo(id, DEVICE_MAX_WORK_GROUP_SIZE, sizeof(int), addr result, nil)

proc localMemory*(id: PDeviceId): uint64 =
  check getDeviceInfo(id, DEVICE_LOCAL_MEM_SIZE, sizeof(int), addr result, nil)

proc globalMemory*(id: PDeviceId): uint64 =
  check getDeviceInfo(id, DEVICE_GLOBAL_MEM_SIZE, sizeof(int), addr result, nil)

proc maxWorkItems*(id: PDeviceId): seq[int] =
  var dims: int
  check getDeviceInfo(id, DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(int), addr dims, nil)
  result = newSeq[int](dims)
  check getDeviceInfo(id, DEVICE_MAX_WORK_ITEM_SIZES, dims * sizeof(int), addr result[0], nil)

proc version*(id: PPlatformId): string =
  var size = 0
  check getPlatformInfo(id, PLATFORM_VERSION, 0, nil, addr size)
  result = newString(size)
  check getPlatformInfo(id, PLATFORM_VERSION, size, addr result[0], nil)

proc getPlatformByName*(platformName: string): PPlatformId =
  var numPlatforms: uint32
  check getPlatformIDs(0, nil, addr numPlatforms)
  var platforms = newSeq[PPlatformId](numPlatforms)
  check getPlatformIDs(numPlatforms, addr platforms[0], nil)

  for platform in platforms:
    if platform.name.substr(0, platformName.high) == platformName:
      return platform

  raise newPlatformNotFound()

proc firstPlatform*(): PPlatformId =
  var numPlatforms: uint32
  check getPlatformIDs(0, nil, addr numPlatforms)
  if numPlatforms == 0:
    raise newPlatformNotFound()
  var platforms = newSeq[PPlatformId](numPlatforms)
  check getPlatformIDs(numPlatforms, addr platforms[0], nil)
  return platforms[0]

proc getDevices*(platform: PPlatformId): seq[PDeviceId] =
  var numDevices: uint32
  check getDeviceIDs(platform, DEVICE_TYPE_ALL, 0, nil, addr numDevices)
  if numDevices == 0:
    raise newDeviceNotFound()

  var devices = newSeq[PDeviceId](numDevices)
  check getDeviceIDs(platform, DEVICE_TYPE_ALL, numDevices, addr devices[0], nil)
  devices

proc createContext*(devices: seq[PDeviceId]): PContext =
  var status: TClResult
  var devs = devices
  result = createContext(nil, devs.len.uint32, cast[ptr PDeviceId](addr devs[0]), nil, nil, addr status)
  check status

proc commandQueueFor*(context: PContext, device: PDeviceId): PCommandQueue =
  var status: TClResult
  result = createCommandQueue(context, device, 0, addr status)
  check status

proc openclDefaults*(): tuple[devices: seq[PDeviceId], context: PContext] =
  let
    platform = firstPlatform()
    devices = platform.getDevices
    context = devices.createContext
  return (devices, context)

proc singleDeviceDefaults*(): tuple[device: PDeviceId, context: PContext, queue: PCommandQueue] =
  let
    platform = firstPlatform()
    device = platform.getDevices[0]
    context = @[device].createContext
    queue = context.commandQueueFor(device)
  return (device, context, queue)

proc createProgram*(context: PContext, body: string): PProgram =
  var status: TClResult
  var lines = [cstring(body)]
  result = createProgramWithSource(context, 1, cast[cstringArray](addr lines), nil, addr status)
  check status

proc buildOn*(program: PProgram, devices: seq[PDeviceId]) =
  var devs = devices
  check buildProgram(program, devs.len.uint32, cast[ptr PDeviceId](addr devs[0]), nil, nil, nil)

proc buildOn*(program: PProgram, device: PDeviceId) = program.buildOn(@[device])

proc createAndBuild*(context: PContext, body: string, devices: seq[PDeviceId]): PProgram =
  result = createProgram(context, body)
  result.buildOn(devices)

proc createAndBuild*(context: PContext, body: string, device: PDeviceId): PProgram =
  result = createProgram(context, body)
  result.buildOn(device)

proc buffer*[A](context: PContext, size: int, flags: Tmem_flags = MEM_READ_WRITE): PMem =
  var status: TClResult
  result = createBuffer(context, flags, size * sizeof(A), nil, addr status)
  check status

proc bufferLike*[A](context: PContext, xs: seq[A], flags: Tmem_flags = MEM_READ_WRITE): PMem =
  buffer[A](context, xs.len, flags)

proc buildErrors*(program: PProgram, devices: seq[PDeviceId]): string =
  var logSize: int
  check getProgramBuildInfo(program, devices[0], PROGRAM_BUILD_LOG, 0, nil, addr logSize)
  result = newString(logSize + 1)
  check getProgramBuildInfo(program, devices[0], PROGRAM_BUILD_LOG, logSize, addr result[0], nil)

proc createKernel*(program: PProgram, name: string): PKernel =
  var status: TClResult
  result = createKernel(program, name, addr status)
  check status

type
  LocalBuffer*[A] = distinct int
  anyInt = int or int32 or int64

template setArg(kernel: PKernel, item: PMem, index: int) =
  var x = item
  check setKernelArg(kernel, index.uint32, sizeof(Pmem), addr x)

template setArg[A](kernel: PKernel, item: var A, index: int) =
  check setKernelArg(kernel, index.uint32, sizeof(A), addr item)

template setArg[A](kernel: PKernel, item: LocalBuffer[A], index: int) =
  check setKernelArg(kernel, index.uint32, int(item) * sizeof(A), nil)

template setArg(kernel: PKernel, item: anyInt, index: int) =
  var x = item
  check setKernelArg(kernel, index.uint32, sizeof(type(item)), addr x)

proc args*[A1](kernel: PKernel, a1: A1) =
  kernel.setArg(a1, 0)

proc args*[A1, A2](kernel: PKernel, a1: A1, a2: A2) =
  kernel.setArg(a1, 0)
  kernel.setArg(a2, 1)

proc args*[A1, A2, A3](kernel: PKernel, a1: A1, a2: A2, a3: A3) =
  kernel.setArg(a1, 0)
  kernel.setArg(a2, 1)
  kernel.setArg(a3, 2)

proc args*[A1, A2, A3, A4](kernel: PKernel, a1: A1, a2: A2, a3: A3, a4: A4) =
  kernel.setArg(a1, 0)
  kernel.setArg(a2, 1)
  kernel.setArg(a3, 2)
  kernel.setArg(a4, 3)

proc args*[A1, A2, A3, A4, A5](kernel: PKernel, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5) =
  kernel.setArg(a1, 0)
  kernel.setArg(a2, 1)
  kernel.setArg(a3, 2)
  kernel.setArg(a4, 3)
  kernel.setArg(a5, 4)

proc args*[A1, A2, A3, A4, A5, A6](kernel: PKernel, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5, a6: A6) =
  kernel.setArg(a1, 0)
  kernel.setArg(a2, 1)
  kernel.setArg(a3, 2)
  kernel.setArg(a4, 3)
  kernel.setArg(a5, 4)
  kernel.setArg(a6, 5)

proc run*(queue: PCommandQueue, kernel: PKernel, totalWork: int) =
  var globalWorkSize = [totalWork, 0, 0]
  check enqueueNDRangeKernel(queue, kernel, 1, nil,  cast[ptr int](addr globalWorkSize), nil, 0, nil, nil)

proc run*(queue: PCommandQueue, kernel: PKernel, totalWork, localWork: int) =
  var
    globalWorkSize = [totalWork, 0, 0]
    localWorkSize = [localWork, 0, 0]
  check enqueueNDRangeKernel(queue, kernel, 1, nil,  cast[ptr int](addr globalWorkSize), cast[ptr int](addr localWorkSize), 0, nil, nil)

proc run2d*(queue: PCommandQueue, kernel: PKernel, totalWork: (int, int)) =
  let (a, b) = totalWork
  var globalWorkSize = [a, b, 0]
  check enqueueNDRangeKernel(queue, kernel, 1, nil,  cast[ptr int](addr globalWorkSize), nil, 0, nil, nil)

proc run2d*(queue: PCommandQueue, kernel: PKernel, totalWork, localWork: (int, int)) =
  let
    (a, b) = totalWork
    (c, d) = localWork
  var
    globalWorkSize = [a, b, 0]
    localWorkSize = [c, d, 0]
  check enqueueNDRangeKernel(queue, kernel, 1, nil,  cast[ptr int](addr globalWorkSize), cast[ptr int](addr localWorkSize), 0, nil, nil)

proc run3d*(queue: PCommandQueue, kernel: PKernel, totalWork: (int, int, int)) =
  let (a, b, c) = totalWork
  var globalWorkSize = [a, b, c]
  check enqueueNDRangeKernel(queue, kernel, 1, nil,  cast[ptr int](addr globalWorkSize), nil, 0, nil, nil)

proc run3d*(queue: PCommandQueue, kernel: PKernel, totalWork, localWork: (int, int, int)) =
  let
    (a, b, c) = totalWork
    (d, e, f) = localWork
  var
    globalWorkSize = [a, b, c]
    localWorkSize = [d, e, f]
  check enqueueNDRangeKernel(queue, kernel, 1, nil,  cast[ptr int](addr globalWorkSize), cast[ptr int](addr localWorkSize), 0, nil, nil)

proc write*(queue: PCommandQueue, src: pointer, dest: PMem, size: int) =
  check enqueueWriteBuffer(queue, dest, CL_FALSE, 0, size, src, 0, nil, nil)

proc write*[A](queue: PCommandQueue, src: var seq[A], dest: PMem) =
  write(queue, addr src[0], dest, src.len * sizeof(A))

proc read*(queue: PCommandQueue, dest: pointer, src: PMem, size: int) =
  check enqueueReadBuffer(queue, src, CL_TRUE, 0, size, dest, 0, nil, nil)

proc read*[A](queue: PCommandQueue, dest: var seq[A], src: PMem) =
  read(queue, addr dest[0], src, dest.len * sizeof(A))

template release*(queue: PCommandQueue) = check releaseCommandQueue(queue)
template release*(kernel: PKernel) = check releaseKernel(kernel)
template release*(program: PProgram) = check releaseProgram(program)
template release*(buffer: PMem) = check releaseMemObject(buffer)
template release*(context: PContext) = check releaseContext(context)