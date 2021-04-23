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
    EPSILON = "EPSILON"
    SEND_FINAL_RESULT = "SEND_FINAL_RESULT"
    MAX_ITERATIONS = "MAX_ITERATIONS"
    MAX_DIMENSIONS = "MAX_DIMENSIONS"
    CENTER = "CENTER"
    SCALE_VARIANCE = "SCALE_VARIANCE"
    LOG2 = "LOG2"
    FEDERATED_QR = "FEDERATED_QR"


class PcaLocalParameter:
    pass


class PcaGlobalParameter:
    pass