"""
    server-side Pca project to aggregate the local parameters from the clients

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

from hyfed_server.project.hyfed_server_project import HyFedServerProject
from hyfed_server.util.hyfed_steps import HyFedProjectStep
from hyfed_server.util.status import ProjectStatus
from hyfed_server.util.utils import client_parameters_to_list

from pca_server.util.pca_steps import PcaProjectStep
from pca_server.util.pca_parameters import PcaGlobalParameter, PcaLocalParameter,\
    PcaProjectParameter
from pca_server.util.pca_algorithms import PcaAlgorithm
import numpy as np


import logging
logger = logging.getLogger(__name__)


class PcaServerProject(HyFedServerProject):
    """ Server side of Pca project """

    def __init__(self, creation_request, project_model):
        """ Initialize Pca project attributes based on the values set by the coordinator """

        # initialize base project
        super().__init__(creation_request, project_model)

        try:
            pca_model_instance = project_model.objects.get(id=self.project_id)

            # attributes
            max_iterations = int(creation_request.data[PcaProjectParameter.MAX_ITERATIONS])
            pca_model_instance.max_iterations = max_iterations
            self.max_iterations = max_iterations

            max_dimensions = int(creation_request.data[PcaProjectParameter.MAX_DIMENSIONS])
            pca_model_instance.max_dimensions = max_dimensions
            self.max_dimensions = max_dimensions

            center = bool(creation_request.data[PcaProjectParameter.CENTER])
            pca_model_instance.center = center
            self.center = center

            scale_variance = bool(creation_request.data[PcaProjectParameter.SCALE_VARIANCE])
            pca_model_instance.scale_variance = scale_variance
            self.scale_variance = scale_variance

            log2 = bool(creation_request.data[PcaProjectParameter.LOG2])
            pca_model_instance.log2 = log2
            self.log2 = log2

            federated_qr = bool(creation_request.data[PcaProjectParameter.FEDERATED_QR])
            pca_model_instance.federated_qr = federated_qr
            self.federated_qr = federated_qr

            send_final_result = bool(creation_request.data[PcaProjectParameter.SEND_FINAL_RESULT])
            pca_model_instance.send_final_result = send_final_result
            self.send_final_result = send_final_result

            epsilon = float(creation_request.data[PcaProjectParameter.EPSILON])
            pca_model_instance.epsilon = epsilon
            self.epsilon = epsilon

            self.current_iteration = 1
            pca_model_instance.current_iteration = 1

            results_dir = 'pca_server/result'
            #pca_model_instance.result_dir = results_dir
            self.result_dir = results_dir
            logger.info(pca_model_instance)

            pca_model_instance.save()

            self.current_iteration = 1
            pca_model_instance.current_iteration = 1

            # init some other global attributes
            self.row_count = -1
            self.column_count = -1

            logger.info(f"Project {self.project_id}: PCA specific attributes initialized!")

        except Exception as model_exp:
            logger.error(model_exp)
            self.project_failed()

    # ############### Project step functions ####################
    def init_step(self):
        """ initialize Pca server project """

        try:
            self.prepare_results()

            self.set_step(HyFedProjectStep.RESULT)

        except Exception as init_exception:
            logger.error(f'Project {self.project_id}: {init_exception}')
            self.project_failed()

    def prepare_results(self):
        """ Prepare result files for Pca project """

        try:
            self.create_result_dir()

        except Exception as io_error:
            logger.error(f"Result file write error: {io_error}")
            self.project_failed()

    # ##############  Pca specific aggregation code
    def aggregate(self):
        """ OVERRIDDEN: perform Pca project specific aggregations """

        # The following four lines MUST always be called before the aggregation starts
        super().pre_aggregate()
        if self.status != ProjectStatus.AGGREGATING:  # if project failed or aborted, skip aggregation
            super().post_aggregate()
            return

        logger.info(f'Project {self.project_id}: ############## aggregate ####### ')
        logger.info(f'Project {self.project_id}: #### step {self.step}')

        if self.step == HyFedProjectStep.INIT:  # The first step name MUST always be HyFedProjectStep.INIT
            self.init_step()

        elif self.step == HyFedProjectStep.RESULT:
            super().result_step()

        # The following line MUST be the last function call in the aggregate function
        super().post_aggregate()
