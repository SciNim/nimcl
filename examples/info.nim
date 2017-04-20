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

import nimcl

proc main() =
  let platform = firstPlatform()
  echo "Using Open CL version:"
  echo "  ", platform.version
  echo "Found platform:"
  echo "  ", platform.name
  let devices = platform.getDevices
  echo "Found ", devices.len, " devices:"
  for device in devices:
    echo "  ", device.name
    echo "  Max work groups for device: " & $(device.maxWorkGroups)
    echo "  Max work items per group: " & $(device.maxWorkItems)
    echo "  Global memory: " & $(device.globalMemory) & " bytes"
    echo "  Local memory: " & $(device.localMemory) & " bytes"

when isMainModule:
  main()