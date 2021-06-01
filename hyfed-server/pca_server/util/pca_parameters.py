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
    EPSILON = "epsilon"
    SEND_FINAL_RESULT = "send_final_result"
    MAX_ITERATIONS = "max_iterations"
    MAX_DIMENSIONS = "max_dimensions"
    CENTER = "center"
    SCALE_VARIANCE = "scale_variance"
    LOG2 = "log2"
    FEDERATED_QR = "federated_qr"
    CURRENT_ITERATION = "current_iteration"
    SPEEDUP = 'speedup'
    USE_SMPC = 'use_smpc'


class PcaLocalParameter:
    COLUMN_COUNT = "COLUMN_COUNT"
    COLUMN_NAMES = "COLUMN_NAMES"
    K = "K"
    ROW_NAMES = "ROW_NAMES"
    ROW_COUNT = "ROW_COUNT"

    GI_MATRIX = "GI_MATRIX"
    LOCAL_DATA = "LOCAL_DATA"
    HI_MATRIX = "HI_MATRIX"
    DELTAS = "DELTAS"
    LOCAL_SUM = "LOCAL_SUM"
    LOCAL_SUM_OF_SQUARES = "LOCAL_SUM_OF_SQUARES"


    LOCAL_CONORMS = "LOCAL_CONORMS"
    LOCAL_EIGENVECTOR_NORM = "LOCAL_EIGENVECTOR_NORM"

    CURRENT_VECTOR_INDEX = "CURRENT_VECTOR_INDEX"
    CURRENT_EIGENVECTOR = "CURRENT_EIGENVECTOR"
    RESULT = "RESULT"


class PcaGlobalParameter:
    GLOBAL_CONORMS = "GLOBAL_CORNOMS"
    GLOBAL_EIGENVECTOR_NORM = "GLOBAL_EIGENVECTOR_NORM"
    GLOBAL_MEANS = "GLOBAL_MEANS"
    GLOBAL_VARIANCE = "GLOBAL_VARIANCE"
    ROW_NAMES = "ROW_NAMES"
    ROW_COUNT = "ROW_COUNT"
    K = "K"
    HI_MATRIX = "HI_MATRIX"
    GI_MATRIX = "GI_MATRIX"
    DELTAS = "DELTAS"
    CURRENT_VECTOR_INDEX = "CURRENT_VECTOR_INDEX"
    CURRENT_EIGENVECTOR = "CURRENT_EIGENVECTOR"