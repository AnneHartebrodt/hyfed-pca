"""
    Pca specific project steps

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


class PcaProjectStep:
    INIT_POWER_ITERATION = "INIT_POWER_ITERATION"
    GLOBAL_MEAN_AND_VAR = "GLOBAL_MEAN_AND_VAR"
    CENTER_AND_SCALE_DATA = "CENTER_AND_SCALE_DATA"
    COMPUTE_H_AND_POOL = "COMPUTE_H_AND_POOL"
    COMPUTE_G_AND_ORTHO = "COMPUTE_G_AND_ORTHO"
    PROJECTIONS = "PROJECTIONS"
    ORTHONORMALISE_EIGENVECTOR_NORM = "ORTHONORMALISE_EIGENVECTOR_NORM"
    ORTHONORMALISE_CONORMS = "ORTHONORMALISE_CONORMS"
    INIT_NEXT_EIGENVECTOR = "INIT_NEXT_EIGENVECTOR"
    RESULTS = "RESULTS"
    COMPUTE_H_NOT_G = "COMPUTE_H_NOT_G"