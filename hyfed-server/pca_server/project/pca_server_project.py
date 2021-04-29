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
import scipy.linalg as la
import scipy as sc
import pandas as pd


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
            self.k = max_dimensions

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
            self.converged = False
            self.current_vector_index = 0
            self.orthonormalisation_done = False
            self.deltas = []

            logger.info(f"Project {self.project_id}: PCA specific attributes initialized!")

        except Exception as model_exp:
            logger.error(model_exp)
            self.project_failed()

    # ############### Project step functions ####################
    def init_step(self):
        """ initialize Pca server project """

        try:
            client_rows = client_parameters_to_list(self.local_parameters, PcaLocalParameter.ROW_NAMES)
            self.count_rows_and_names(client_rows)
            client_columns = client_parameters_to_list(self.local_parameters, PcaLocalParameter.COLUMN_NAMES)
            self.count_columns_and_names(client_columns)
            self.set_step(PcaProjectStep.INIT_POWER_ITERATION)

            self.global_parameters[PcaLocalParameter.ROW_NAMES] = self.get_row_names()
            self.global_parameters[PcaLocalParameter.K] = self.k

            #self.prepare_results()
            #self.set_step(HyFedPcaProjectStep.RESULT)

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

    def save_pcs(self):
        parameters_from_clients = client_parameters_to_list(self.local_parameters,
                                                            PcaLocalParameter.GI_MATRIX)
        gis = []
        for client_parameter in parameters_from_clients:
            gis.append(client_parameter)
        self.g_matrix = np.concatenate(gis, axis=0)

        try:
            logger.info("Writing pcs to file ...")
            pd.DataFrame(self.column_names).to_csv(self.result_file_path+'.varnames', header=False, index=False, sep='\t')
            pd.DataFrame(self.g_matrix).to_csv(self.result_file_path, header=False, index=False, sep='\t')
            logger.info("Writing results done!")

        except Exception as exception:
            logger.debug(f"{exception}")

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
        logger.info(f'Project {self.project_id}: #### step {self.local_parameters}')

        if self.step == HyFedProjectStep.INIT:  # The first step name MUST always be HyFedPcaProjectStep.INIT
            self.init_step()

        elif self.step == PcaProjectStep.GLOBAL_MEAN_AND_VAR:
            self.compute_global_mean_and_variance()
            self.set_step(PcaProjectStep.CENTER_AND_SCALE_DATA)
            self.global_parameters[PcaGlobalParameter.GLOBAL_MEANS] = self.get_global_means()
            self.global_parameters[PcaGlobalParameter.GLOBAL_VARIANCE] = self.get_global_variance()

        elif self.step == PcaProjectStep.INIT_POWER_ITERATION:
            self.init_power_iteration()
            self.H_aggregation()
            # here branch into either global QR or local QR
            if not self.federated_qr:
                self.set_step(PcaProjectStep.COMPUTE_G_AND_ORTHO)
            else:
                # first round goes without the projection step,
                # directly get the local norms for the clients
                self.set_step(PcaProjectStep.ORTHONORMALISE_EIGENVECTOR_NORM)
            # current vector index only relevant for QR
            self.global_parameters[PcaGlobalParameter.HI_MATRIX] = self.get_global_h_matrix()
            self.global_parameters[PcaGlobalParameter.CURRENT_VECTOR_INDEX] = self.get_current_vector_index()
            self.global_parameters[PcaGlobalParameter.DELTAS] = self.deltas

        elif self.step == PcaProjectStep.COMPUTE_H_AND_POOL or \
                self.step == PcaProjectStep.CENTER_AND_SCALE_DATA:
            self.H_aggregation()
            if self.current_iteration % 10 == 0:
                if self.federated_qr:
                    self.set_step(PcaProjectStep.ORTHONORMALISE_EIGENVECTOR_NORM)
                else:
                    self.set_step(PcaProjectStep.COMPUTE_G_AND_ORTHO)
            else:
                self.set_step(PcaProjectStep.COMPUTE_H_NOT_G)

            self.global_parameters[PcaGlobalParameter.HI_MATRIX] =  self.get_global_h_matrix()
            self.global_parameters[PcaGlobalParameter.CURRENT_VECTOR_INDEX]= self.get_current_vector_index()
            self.global_parameters[PcaGlobalParameter.DELTAS] = self.deltas

        elif self.step == PcaProjectStep.COMPUTE_H_NOT_G:
            self.H_aggregation()
            if self.current_iteration % 10 == 0 or self.converged:
                self.set_step(PcaProjectStep.ORTHONORMALISE_EIGENVECTOR_NORM)
            else:
                self.set_step(PcaProjectStep.COMPUTE_H_NOT_G)


        elif self.step == PcaProjectStep.ORTHONORMALISE_CONORMS:
            self.aggregate_conorms()
            self.set_step(PcaProjectStep.ORTHONORMALISE_EIGENVECTOR_NORM)

            self.global_parameters[PcaGlobalParameter.GLOBAL_CONORMS] = self.get_global_conorms()
            self.global_parameters[PcaGlobalParameter.CURRENT_VECTOR_INDEX] = self.get_current_vector_index()



        elif self.step == PcaProjectStep.ORTHONORMALISE_EIGENVECTOR_NORM:
            self.aggregate_eigenvector_norms()
            logger.info('Current vector index' + str(self.get_current_vector_index()))
            if self.orthonormalisation_done:
                self.reset_current_vector_index()
                logger.info('Current vector index- ortho done' + str(self.get_current_vector_index()))
                if not self.converged:
                    self.set_step(PcaProjectStep.COMPUTE_H_AND_POOL)
                    self.global_parameters[PcaGlobalParameter.CURRENT_VECTOR_INDEX] = self.get_current_vector_index()
                    self.global_parameters[PcaGlobalParameter.GLOBAL_EIGENVECTOR_NORM] = self.get_global_eigenvector_norm()

                else:
                    self.set_step(PcaProjectStep.RESULTS)
                    self.global_parameters[PcaGlobalParameter.GLOBAL_EIGENVECTOR_NORM] = self.get_global_eigenvector_norm()
                    self.global_parameters[PcaGlobalParameter.CURRENT_VECTOR_INDEX] = self.get_current_vector_index()

            else:

                self.set_step(PcaProjectStep.ORTHONORMALISE_CONORMS)
                self.global_parameters[PcaGlobalParameter.GLOBAL_EIGENVECTOR_NORM] = self.get_global_eigenvector_norm()
                self.global_parameters[PcaGlobalParameter.CURRENT_VECTOR_INDEX] = self.get_current_vector_index()


        # elif self.step == PcaProjectStep.COMPUTE_G_AND_ORTHO:
        #     self.G_orthonormalisation(parameters_from_clients, client_ids)
        #     if not self.converged:
        #         self.project_step = PcaProjectStep.COMPUTE_H_AND_POOL
        #         param_obj = []
        #         for g in self.g_matrices:
        #             logger.debug(g.shape)
        #             param_obj.append({PcaGlobalParameter.PROJECT_STEP: self.step,
        #                               PcaGlobalParameter.PROJECT_STATUS: self.get_project_status(),
        #                               PcaGlobalParameter.GI_MATRIX: g,
        #                               })
        #         self.param_obj = param_obj
        #         return False
        #     else:
        #         if self.project_status == ProjectStatus.RUNNING:
        #             self.project_status = ProjectStatus.DONE
        #         self.project_step = PcaProjectStep.RESULTS
        #         if not self.send_final_result:
        #             param_obj = []
        #             # only send partial vectors
        #             for g in self.g_matrices:
        #                 param_obj.append({PcaGlobalParameter.PROJECT_STEP: self.step,
        #                                   PcaGlobalParameter.PROJECT_STATUS: self.get_project_status(),
        #                                   PcaGlobalParameter.GI_MATRIX: g,
        #                                   })
        #             self.param_obj = param_obj
        #         else:
        #             # send full vectors
        #             self.g_matrix = np.concatenate(self.g_matrices, axis=1)
        #             self.param_obj = {PcaGlobalParameter.PROJECT_STEP: self.step,
        #                               PcaGlobalParameter.PROJECT_STATUS: self.get_project_status(),
        #                               PcaGlobalParameter.GI_MATRIX: self.g_matrix,
        #                               }
        #
        #         return True

        elif self.step == PcaProjectStep.RESULTS:
            super().result_step()

            # if self.federated_qr:
            #     # this recieves and gets the parameters from the clients.
            #     self.save_pcs()

            # WEBSITE STUFF. EIther create here independently from self.project
            ## OR do something else
            # plot_thread = threading.Thread(target=self.project.create_scatter_plot, args=())
            # plot_thread.start()

        elif self.step == HyFedProjectStep.RESULT:
            super().result_step()


        # The following line MUST be the last function call in the aggregate function
        super().post_aggregate()


    def init_power_iteration(self, client_ids=None):
        self.current_iteration = 1
        # Centralised QR aggregation requires the client matrices to be in the same order
        #TODO check this!
        #self.fixed_client_order = client_ids

        def generate_random_gaussian(n, m):
            draws = n * m
            noise = sc.random.normal(0, 1, draws)
            print('Generated random intial matrix: finished sampling')
            # make a matrix out of the noise
            noise.shape = (n, m)
            # transform matrix into s
            return noise

        hi = generate_random_gaussian(self.row_count, self.k)
        hi, R = la.qr(hi, mode='economic')
        self.global_h_matrix = hi

        self.current_vector_index = 0

    def count_columns_and_names(self, parameters_from_clients):

        logger.info("Agreeing on common columns ...")
        try:
            all_column_names = []
            for client_parameter in parameters_from_clients:
                all_column_names = np.concatenate([all_column_names,client_parameter])
            self.column_count = len(all_column_names)
            self.column_names = np.array(all_column_names)
            # k can maximally have the smaller dimension of any of the matrices
            km = np.max([len(l) for l in parameters_from_clients])
            self.k = min(self.k, km)
            logger.info('K' + str(self.k))

        except Exception as exception:
            self.project_failed()
            logger.debug(f'{exception}')

    def count_rows_and_names(self, parameters_from_clients):
        '''
        PCA is only calculated using common variables from all clients
        Therefore we send the intersection of all variables back.
        :param parameters_from_clients:
        :return: the intersection of all client variable names
        '''
        logger.info("Agreeing on common parameters ...")
        try:
            next_client = parameters_from_clients[0]
            intersect_column_names = next_client
            for client_parameter in parameters_from_clients:
                next_client = client_parameter
                intersect_column_names = set(intersect_column_names).intersection(next_client)

            self.row_names = np.array(list(intersect_column_names))
            self.row_count = len(self.row_names)
            self.k = min(self.k, self.row_count)
            logger.info('K' + str(self.k))
            logger.info("Found common row names")


        except Exception as exception:
            self.project_failed()
            logger.error(f'{exception}')


    ## server aggreation
    def H_aggregation(self):
        '''
        This step adds up the H matrices (nr_SNPs x target dimension) matrices to
        achieve a global H matrix
        :param parameters_from_clients: The local H matrices
        :return: Global H matrix to be sent to the client
        '''
        logger.info("Adding up H matrices from clients ...")

        try:
            parameters_from_clients = client_parameters_to_list(self.local_parameters, PcaLocalParameter.HI_MATRIX)
            self.current_iteration += 1
            global_HI_matrix = None
            for client_parameter in parameters_from_clients:
                try:
                    if global_HI_matrix is not None:
                        global_HI_matrix += client_parameter
                    else:
                        global_HI_matrix = client_parameter
                except:
                    self.project_failed()

            global_HI_matrix, R = la.qr(global_HI_matrix, mode='economic')
            logger.info('global_hi' + str(global_HI_matrix.shape))
            # The previous H matrix is stored in the global variable
            if self.current_iteration == self.max_iterations\
                    or self.eigenvector_convergence_checker(global_HI_matrix, self.global_h_matrix, self.epsilon):
                self.converged = True
                logger.info('converged' + str(self.converged))

            self.global_h_matrix = global_HI_matrix
            logger.info(self.global_h_matrix.shape)
            logger.info("H matrix step aggregation done!")


        except Exception as exception:
            self.project_failed()
            logger.debug(f"{exception}")

    def eigenvector_convergence_checker(self, current, previous, tolerance=PcaProjectParameter.EPSILON):
        '''

        Args:
            current: The current eigenvector estimate
            previous: The eigenvector estimate from the previous iteration
            tolerance: The error tolerance for eigenvectors to be equal
            required: optional parameter for the number of eigenvectors required to have converged

        Returns: True if the required numbers of eigenvectors have converged to the given precision, False otherwise

        '''
        nr_converged = 0
        col = 0
        converged = False
        required = current.shape[1]
        deltas = []
        while col < current.shape[1] and not converged:
            # check if the scalar product of the current and the previous eigenvectors
            # is 1, which means the vectors are 'parallel'
            # print('angle' +str(co.angle(current[:, col], previous[:, col])))
            delta = np.abs(np.sum(np.dot(np.transpose(current[:, col]), previous[:, col])))
            deltas.append(delta)
            if delta >= 1 - tolerance:
                nr_converged = nr_converged + 1
            if nr_converged >= required:
                converged = True
            col = col + 1
        self.deltas = deltas
        return converged

    def G_orthonormalisation(self, parameters_from_clients, client_ids):
        logger.info("Orthonormalisation step ...")

        try:
            for client_id in client_ids:
                logger.info(client_id)
            # this is the order in which the variable arrived
            current_order = {}
            for i in range(len(client_ids)):
                current_order[client_ids[i]] = i
            # create the order that the clients have to assume
            indices = []
            for id in self.fixed_client_order:
                indices.append(current_order[id])

            client_GI_matrices = []
            client_samples_sizes = []
            for i in indices:
                logger.info(parameters_from_clients[i][Parameter.GI_MATRIX].shape)
                client_GI_matrices.append(parameters_from_clients[i][Parameter.GI_MATRIX])
                client_samples_sizes.append(parameters_from_clients[i][Parameter.GI_MATRIX].shape[0])
            # stack the matrices. Now in the order of the clients from init
            global_GI_matrices = np.concatenate(client_GI_matrices, axis = 0)

            G, Q = la.qr(global_GI_matrices, mode='economic')
            self.g_matrix = G

            g_matrices = []
            sum = 0
            # make sure to send the correct indices to the correct client
            for i in indices:
                c = client_samples_sizes[i]
                g_matrices.append(G[sum:sum+c, :])
                sum = sum+c

            self.g_matrices = g_matrices
            logger.info("Orthonormalisation done!")


        except Exception as exception:
            self.project_failed()
            logger.debug(f"{exception}")

    def aggregate_eigenvector_norms(self):
        parameters_from_clients = client_parameters_to_list(self.local_parameters, PcaLocalParameter.LOCAL_EIGENVECTOR_NORM)

        eigenvector_norm = 0
        for client_parameter in parameters_from_clients:
            eigenvector_norm = eigenvector_norm + client_parameter

        self.global_eigenvector_norm = eigenvector_norm
        logger.info('Current vector' + str(self.current_vector_index))
        logger.info('Eigenvector norm'+str(self.global_eigenvector_norm))
        # increment the vector index after sending back the norms
        self.current_vector_index = self.current_vector_index + 1
        if self.get_current_vector_index() == self.k:
            self.orthonormalisation_done = True

    def aggregate_conorms(self):
        parameters_from_clients = client_parameters_to_list(self.local_parameters,
                                                            PcaLocalParameter.LOCAL_CONORMS)
        # conorms will have the length of the local conorms
        conorms = [0]* len(parameters_from_clients[0])
        for client_parameter in parameters_from_clients:
            local_conorms = client_parameter
            for ln in range(len(local_conorms)):
                conorms[ln] += local_conorms[ln]
        self.global_conorms = conorms
        logger.info('Current vector' + str(self.current_vector_index))
        logger.info('Conorms' + str(self.global_conorms))



    def get_column_names(self):
        return self.column_names

    def get_row_names(self):
        return self.row_names

    def get_global_means(self):
        return self.global_means

    def get_global_variance(self):
        return self.global_variance

    def get_global_h_matrix(self):
        return self.global_h_matrix

    def get_g_matrix(self):
        return self.g_matrix

    def get_global_conorms(self):
        return self.global_conorms

    def get_global_eigenvector_norm(self):
        return self.global_eigenvector_norm

    def get_current_vector_index(self):
        return self.current_vector_index

    def reset_current_vector_index(self):
        self.current_vector_index = 0
        self.orthonormalisation_done = False

    def get_project_step(self):
        return self.project_step

