"""
    Pca specific server and client parameters

    Copyright 2021 'My Name'. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""


class PcaProjectParameter:
    EPSILON = 0.000001
    NULL_VALUE = "NA"
    SEND_FINAL_RESULT = False
    FEDERATED_QR = True
    MAX_ITERATIONS = 500
    MAX_DIMENSIONS = 10
    CENTER = True
    SCALE_VARIANCE = True
    LOG2 = True
    pass


class PcaLocalParameter:
    pass


class PcaGlobalParameter:
    pass
