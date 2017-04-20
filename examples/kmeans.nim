# Copyright 2016-2017 UniCredit S.p.A.
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

import times, json, os, math, strutils, opencl, nimcl, point

proc loadPoints(filename: string): seq[Point] =
  result = newSeq[Point]()
  for p in parseFile(filename).items:
    result.add(Point(x: p[0].fnum, y: p[1].fnum, cluster: -1))

proc main() =
  const
    body = staticRead("kmeans.cl")
    n = 10
    iterations = 100
  var
    points = loadPoints("points.json")
    centroids = newSeq[Centroid](n)
  let
    (device, context, queue) = singleDeviceDefaults()
    program = context.createProgram(body)
    workGroups = device.maxWorkGroups
    workItems = (points.len div workGroups).nextPowerOfTwo

  program.buildOn(device)

  let
    groupByCluster = program.createKernel("group_by_cluster")
    sumPoints = program.createKernel("sum_points")
    updateCentroids = program.createKernel("update_centroids")
    start = cpuTime()
    gpuPoints = context.bufferLike(points)
    gpuCentroids = context.bufferLike(centroids)
    gpuAccum = buffer[Accum](context, centroids.len * workGroups)

  groupByCluster.args(gpuPoints, gpuCentroids, points.len.int32, centroids.len.int32)
  sumPoints.args(gpuPoints, gpuAccum, LocalBuffer[Accum](centroids.len * workItems), points.len.int32, centroids.len.int32)
  updateCentroids.args(gpuAccum, gpuCentroids, workGroups.int32, centroids.len.int32)

  for _ in 1 .. iterations:
    for i in 0 .. < centroids.len:
      centroids[i].x = points[i].x
      centroids[i].y = points[i].y

    queue.write(points, gpuPoints)
    queue.write(centroids, gpuCentroids)

    for _ in 1 .. 15:
      queue.run(groupByCluster, points.len)
      queue.run(sumPoints, workItems * workGroups, workItems)
      queue.run(updateCentroids, centroids.len)

    queue.read(centroids, gpuCentroids)

  let time = (((cpuTime() - start) * 1000) / float(iterations)).round
  echo format("Made $1 iterations with an average of $2 milliseconds",
              iterations, time)

  for a in centroids:
    echo a

  # Clean up
  release(queue)
  release(groupByCluster)
  release(sumPoints)
  release(updateCentroids)
  release(program)
  release(gpuPoints)
  release(gpuCentroids)
  release(context)

when isMainModule:
  main()