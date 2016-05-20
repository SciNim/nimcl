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

import os, nimcl

proc main() =
  let n = paramCount()
  if n != 1:
    echo "Please, use exactly one argument"
    return
  let
    fileName = paramStr(1)
    body = readFile(fileName)
    platform = getPlatformByName("NVIDIA CUDA")
    devices = platform.getDevices
    context = devices.createContext
    program = context.createProgram(body)
  try:
    program.buildOn(devices)
    echo "Program compiled"
  except:
    echo "Build failure"
    echo program.buildErrors(devices)

when isMainModule:
  main()