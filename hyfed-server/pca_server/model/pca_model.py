"""
    Model class for Pca project specific fields

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

from django.db import models
from hyfed_server.model.hyfed_models import HyFedProjectModel


class PcaProjectModel(HyFedProjectModel):
    """
        The model inherits from HyFedProjectModel
        so it implicitly has id, name, status, etc, fields defined in the parent model
    """
    max_iterations = models.PositiveIntegerField(default=500)
    max_dimensions = models.PositiveIntegerField(default=2)
    center = models.BooleanField(default=False)
    scale_variance = models.BooleanField(default=False)
    log2 = models.BooleanField(default=False)
    federated_qr = models.BooleanField(default=True)
    send_final_result = models.BooleanField(default=False)
    current_iteration = models.PositiveIntegerField(default=1)
    epsilon = models.FloatField(default=1e-9)
    speedup = models.BooleanField(default=True)
    use_smpc = models.BooleanField(default=True)

