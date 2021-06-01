"""
    Client-side Pca project to compute local parameters

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

from hyfed_client.project.hyfed_client_project import HyFedClientProject
from hyfed_client.util.hyfed_steps import HyFedProjectStep

from pca_client.util.pca_steps import PcaProjectStep
from pca_client.util.pca_parameters import PcaGlobalParameter, PcaLocalParameter

import pandas as pd
import numpy as np
import scipy as sc

class PcaClientProject(HyFedClientProject):
    """
        A class that provides the computation functions to compute local parameters
    """

    def __init__(self, username, token, project_id, server_url, compensator_url,
                 tool, algorithm, name, description, coordinator, result_dir, log_dir,
                 pca_datasets_file_path, max_iterations, max_dimensions, center,
                 scale_variance, log2, federated_qr, send_final_result, speedup, has_rownames,
                 has_column_names, field_delimiter, use_smpc):

        super().__init__(username=username, token=token, project_id=project_id, server_url=server_url,
                         compensator_url=compensator_url, tool=tool,
                         algorithm=algorithm, name=name, description=description, coordinator=coordinator,
                         result_dir=result_dir, log_dir=log_dir)

        # global settings/parameters
        self.max_iterations = max_iterations
        self.max_dimensions = max_dimensions
        self.center = center
        self.scale_variance = scale_variance
        self.log2 = log2
        self.federated_qr = federated_qr
        self.send_final_result = send_final_result
        self.speedup = speedup
        self.use_smpc = use_smpc

        #local settings
        self.pca_dataset_file_path = pca_datasets_file_path
        self.has_rownames = has_rownames
        self.has_column_names = has_column_names
        self.field_delimiter = field_delimiter
        # init some more stuff here maybe?

        # init some helpers
        self.all_global_eigenvector_norms = []
        self.current_vector_index = 0
        self.orthonormalisation_done = False

        # Default settings
        self.mean_impute = True
        self.impute_neg_inf = True


    # ########## Pca step functions
    def init_step(self):
        # OPEN DATASET FILE(S) AND INITIALIZE THEIR CORRESPONDING DATASET ATTRIBUTES
        try:
            self.read_file()
            self.sample_feature_count_step()
            # TODO: implement center step
            center = False
            # if center:
            #     self.set_step(PcaProjectStep.GLOBAL_MEAN_AND_VAR)
            # else:
            #     self.set_step(PcaProjectStep.INIT_POWER_ITERATION)

        except Exception as io_exception:
            self.log(io_exception)
            self.set_operation_status_failed()

    def compute_local_parameters(self):
        """ OVERRIDDEN: Compute the local parameters in each step of the Pca algorithms """

        try:

            super().pre_compute_local_parameters()  # MUST be called BEFORE step functions

            # ############## Pca specific local parameter computation steps
            if self.project_step == HyFedProjectStep.INIT:
                self.init_step()

            elif self.project_step == PcaProjectStep.GLOBAL_MEAN_AND_VAR:
                self.remove_invalid_var_names()
                self.compute_local_sums_and_sums_of_squares()

            elif self.project_step == PcaProjectStep.INIT_POWER_ITERATION:
                self.init_power_iteration()
                # Compute H
                self.local_DxG()

            elif self.project_step == PcaProjectStep.CENTER_AND_SCALE_DATA:
                self.center_and_scale()
                self.init_power_iteration()
                # Compute H
                self.local_DxG()

            elif self.project_step == PcaProjectStep.COMPUTE_H_AND_POOL:
                # We recieve the orthogonalised G matrix
                # and compute the local H matrices.
                if self.federated_qr:
                    self.normalise_orthogonalised_matrix()
                    self.local_DxG()

                else:
                    self.local_DxG()

            elif self.project_step == PcaProjectStep.COMPUTE_H_NOT_G:
                # Compute a new H(ope) without othogonalising H
                self.local_DTxH()
                self.local_DxG()

            elif self.project_step == PcaProjectStep.ORTHONORMALISE_CONORMS:
                self.calculate_local_vector_conorms()

            elif self.project_step == PcaProjectStep.ORTHONORMALISE_EIGENVECTOR_NORM:
                # in the first round the new G needs to be computed
                if self.current_vector_index == 0:
                    self.local_DTxH()
                # first orthonormalise (not in the first round)
                if self.global_parameters[PcaLocalParameter.CURRENT_VECTOR_INDEX] > 0:
                    self.orthogonalise_current()

                # then calculate norm
                self.local_eigenvector_norms()
                if self.orthonormalisation_done:
                    self.orthonormalisation_done = False


            # elif self.project_step == PcaProjectStep.COMPUTE_G_AND_ORTHO:
            #     # we recieve the pooled h-matrix and need to to compute
            #     # the local G matrices
            #     self.local_DTxH()
            #     self.requestType = ParameterRequestType.CLIENT_SPECIFIC
            #     self.project_step = PcaProjectStep.COMPUTE_H_AND_POOL

            elif self.project_step == PcaProjectStep.PROJECTIONS:
                self.projection()

            elif self.project_step == PcaProjectStep.RESULTS:
                # Algorithm convergence critoerion has been met
                # we return results if applies
                if self.federated_qr:
                    self.normalise_orthogonalised_matrix()
                self.create_local_result_obj()
            # We jump this step because no global results will be availble
            elif self.project_step == HyFedProjectStep.RESULT:
                super().result_step()  # the result step downloads the result file as zip (it is algorithm-agnostic)
            elif self.project_step == HyFedProjectStep.FINISHED:
                super().finished_step()  # The operations in the last step of the project is algorithm-agnostic

            super().post_compute_local_parameters()  # # MUST be called AFTER step functions
        except Exception as computation_exception:
            self.log(computation_exception)
            super().post_compute_local_parameters()
            self.set_operation_status_failed()


    def read_file(self):
        """

        :return:
        """
        self.log('Reading data ...')
        self.log('Rownames'+ str(self.has_rownames))
        self.log('Column names '+ str(self.has_column_names))
        self.log('Field delimiter '+ str(self.field_delimiter))
        try:
            if self.has_rownames:
                rn = 0
            else:
                rn = None
            if self.has_column_names:
                cn = 0
            else:
                cn = None
            data = pd.read_csv(self.pca_dataset_file_path, header=cn, sep=self.field_delimiter, index_col=rn, engine='python')
        except IOError as exception:
            self.log("Error reading file")
            self.set_operation_status_failed()


        try:
            assert data.shape[1] > 1
            assert data.shape[0] > 1

        except AssertionError as assertion:
            self.log(f"{assertion}")
            self.set_operation_status_failed()


        rownames = data.index.values
        column_names = data.columns.values
        data = data.values

        self.log(rownames)
        self.log(column_names)

        self.data = data
        self.column_names = column_names
        self.rownames = rownames

    def get_column_names(self):
        return self.column_names

    # get the number of samples
    def get_column_count(self):
        try:
            return len(self.column_names)
        except Exception as exception:
            self.log(f"{exception}")
            self.set_operation_status_failed()

    def get_row_count(self):
        try:
            return len(self.rownames)
        except Exception as exception:
            self.log(f"{exception}")
            self.set_operation_status_failed()

    def get_row_names(self):
        try:
            return self.rownames
        except Exception as exception:
            self.log(f"{exception}")
            self.set_operation_status_failed()

    def sample_feature_count_step(self):
        '''
        Transmit the number of samples, the number of features and the feature names
        to the server.
        :return:
        '''

        try:
            self.local_parameters[PcaLocalParameter.ROW_NAMES] = self.get_row_names()
            self.local_parameters[PcaLocalParameter.COLUMN_COUNT] = self.get_column_count()
            self.local_parameters[PcaLocalParameter.COLUMN_NAMES] = self.get_column_names()
            self.local_parameters[PcaLocalParameter.ROW_COUNT] = self.get_row_count()
            self.log(self.local_parameters)

        except Exception as exception:
            self.log(f"\t{exception}\n")
            self.set_operation_status_failed()

    def scale_center_data(self):
        """
        This function centers the data by subtracting the column means. Scaling to equal variance
        is done by dividing the entries by the column standard deviation.
        :param data: nxd numpy array , containing n samples with d variables
        :param center: if true, data is centered by subtracting the column mean
        :param scale_variance: if true, data is scaled by dividing by the standard deviation
        :param impute_neg_inf: set negative infinity produced by taking the log of 0 back to 0
        :param mean_impute: PCA copes badly with missing values, if nan mean impute (recommended)
        :return: the scaled data
        """
        # data.dtype = 'float'
        self.data = self.data.astype('float')
        self.log('Scaling data')
        if self.scale_variance or self.center or self.log2:
            for column in range(self.data.shape[1]):
                mean_val = np.mean(self.data[:, column])
                var_val = np.sqrt(np.var(self.data[:, column], ddof=1))
                # 0 variance columns are not scaled
                if var_val == 0:
                    var_val = 1
                for elem in range(len(self.data[:, column])):
                    if self.center:
                        self.data[elem, column] = self.data[elem, column] - mean_val
                    if self.scale_variance:
                        self.data[elem, column] = self.data[elem, column] / var_val
                    if self.log2:
                        if self.data[elem, column] > 0:
                            self.data[elem, column] = np.log2(self.data[elem, column])
                        elif self.data[elem, column] == 0:
                            if not self.impute_neg_inf:
                                self.data[elem, column] = -np.inf
                            else:
                                self.data[elem, column] = 0
                        else:
                            if self.mean_impute:
                                self.data[elem, column] = mean_val
                            else:
                                # this is not recommended
                                self.data[elem, column] = np.nan

    def remove_invalid_var_names(self):
        '''
        This function reorders the dataframe such that all
        variable names have the same order on all the clients
        and removes variables that are not shared between all
        clients.
        :return:
        '''

        try:

            # in the vertical power iteration, the row names need to be matched
            server_varnames = self.global_parameters[PcaGlobalParameter.ROW_NAMES]
            client_varnames = self.rownames

            self.log('client varnames' + self.rownames)

            # create a dictionay of the positions of the variable
            # names on the server
            server_params = {}
            for i in range(len(server_varnames)):
                server_params[server_varnames[i]] = i

            # identify all client varname positions that need
            # to be dropped
            drop = []
            for i in range(len(client_varnames)):
                if not client_varnames[i] in server_params:
                    drop.append(i)
            # drop varnames from local copy of client varnames
            client_varnames = np.delete(client_varnames, drop)

            # create dictionary of local variable names
            local_var_names = {}
            for i in range(len(client_varnames)):
                local_var_names[client_varnames[i]] = i

            # create the order that the local varnames
            # have to assume
            indices = []
            for i in range(len(server_varnames)):
                indices.append(local_var_names[server_varnames[i]])

            # reassign variable names to object attribute
            self.rownames = client_varnames
            # delete the variable names that need to be deleted
            self.data = np.delete(self.data, drop, axis=0)
            # reorder the local dataframe
            self.data = self.data[indices, :]
            self.log("Removing illegal variable names ... done")
            self.log('client varnames' + self.rownames)

        except Exception as exception:
            self.log(f"\t{exception}\n")
            self.project_failed()

    def center_and_scale(self):
        ''' scaling for vertically partionend data

        In the case of vertical partioning the variables are distributed, meaning
        one client possesses measurements for all samples therefore
        the scaling can be done locally.
        :return: Nothing.
        '''

        try:
            self.scale_center_data()

        except Exception as exception:
            self.log(f"\t{exception}\n")
            self.project_failed()

    def init_power_iteration(self):
        '''
        Just need to generate a correctly formatted inital eigenvector guess
        :return:
        '''

        def generate_random_gaussian(n, m):
            draws = n * m
            noise = sc.random.normal(0, 1, draws)
            # make a matrix out of the noise
            noise.shape = (n, m)
            # transform matrix into s
            return noise

        try:
            self.scale_center_data()
            # retrieve the correct k from the server
            self.k = self.global_parameters[PcaGlobalParameter.K]

            self.g_matrix = generate_random_gaussian(self.get_column_count(), self.k)
            # initial G matrix
            self.global_parameters[PcaGlobalParameter.GI_MATRIX] = self.g_matrix
            self.log('Generate GI' + str(self.get_column_count()) + " " + str(self.k))

        except Exception as exception:
            self.log(f"\t{exception}\n")
            self.project_failed()

    def local_DxG(self):
        try:
            if self.federated_qr:
                Gi = self.get_g_matrix()
            else:
                # Compute dot product of data and G_i
                Gi = self.global_parameters[PcaGlobalParameter.GI_MATRIX]
                self.g_matrix = Gi
            H_i = np.dot(self.data, Gi)

            # self.local_parameters is the object that is sent to the server
            if self.use_smpc:
                self.set_compensator_flag()
            self.local_parameters[PcaLocalParameter.HI_MATRIX] = H_i

        except Exception as exception:
            self.log(f"\t{exception}\n")
            self.project_failed()

    def local_DTxH(self):
        try:

            deltas = self.global_parameters[PcaGlobalParameter.DELTAS]
            self.log('Convergence values' + str(deltas))
            H_i = self.global_parameters[PcaGlobalParameter.HI_MATRIX]
            G_i = self.get_g_matrix()

            G_i_updated = np.dot(self.data.T, H_i) + G_i
            # self.local_parameters is the object that is sent to the server
            self.log('DTxH' + str(G_i_updated.shape))
            self.g_matrix = G_i_updated
            self.h_matrix = H_i
            if self.use_smpc:
                self.set_compensator_flag()
            self.local_parameters[PcaLocalParameter.GI_MATRIX]  = G_i_updated

        except Exception as exception:
            self.log(f"\t{exception}\n")
            self.project_failed()

    def update_H(self):
        try:
            deltas = self.global_parameters[PcaGlobalParameter.DELTAS]
            self.log('Convergence values' + str(deltas))
            H_i = self.global_parameters[PcaGlobalParameter.HI_MATRIX]
            G_i = self.get_g_matrix()

            G_i_updated = np.dot(self.data.T, H_i) + G_i
            H_i_updated = np.dot(self.data, G_i_updated)
            # self.local_parameters is the object that is sent to the server
            self.log('DTxH' + str(G_i_updated.shape))
            self.g_matrix = G_i_updated
            self.h_matrix = H_i
            if self.use_smpc:
                self.set_compensator_flag()
            self.local_parameters[PcaLocalParameter.HI_MATRIX] = H_i_updated

        except Exception as exception:
            self.log(f"\t{exception}\n")
            self.project_failed()

    def calculate_local_vector_conorms(self):
        try:
            g = self.g_matrix
            vector_conorms = []
            # append the lastly calculated norm to the list of global norms
            # we don't need it here
            self.all_global_eigenvector_norms.append(self.global_parameters[PcaGlobalParameter.GLOBAL_EIGENVECTOR_NORM])
            for cvi in range(self.current_vector_index):
                vector_conorms.append(
                    np.dot(g[:, cvi], g[:, self.current_vector_index]) / self.all_global_eigenvector_norms[cvi])
            self.local_vector_conorms = vector_conorms
            self.log('current vector' + str(self.current_vector_index))
            self.log('Local conorms' + str(self.local_vector_conorms))
            if self.current_vector_index == self.k - 1:
                self.orthonormalisation_done = True
            if self.use_smpc:
                self.set_compensator_flag()
            self.local_parameters[PcaLocalParameter.LOCAL_CONORMS]= np.array(self.get_local_vector_conorms())


        except Exception as exception:
            self.log(f"\t{exception}\n")
            self.project_failed()

    def orthogonalise_current(self):
        try:
            self.log('starting orthogonalise_current')
            # global norm
            global_projections = self.global_parameters[PcaGlobalParameter.GLOBAL_CONORMS]
            self.log('Current vector index' + str(self.current_vector_index))
            self.log('Global projections' + str(global_projections))
            # update every cell individually
            for gp in range(len(global_projections)):
                for row in range(self.g_matrix.shape[0]):
                    self.g_matrix[row, self.current_vector_index] = self.g_matrix[row, self.current_vector_index] - \
                                                                    self.g_matrix[row, gp] * global_projections[gp]
            self.log('ending orthogonalise_current')

            # there is a next step!
            # if self.operation_status == OperationStatus.IN_PROGRESS:
            #   self.operation_status = OperationStatus.DONE
        except Exception as exception:
            self.log(f"\t{exception}\n")
            self.project_failed()

    def local_eigenvector_norms(self):
        try:
            self.log('starting eigenvector norms')
            self.log('current_vector_index ' + str(self.current_vector_index))
            # not the euclidean norm, because the square root needs to be calculated
            # at the aggregator
            eigenvector_norm = np.dot(self.get_g_matrix()[:, self.current_vector_index],
                                      self.get_g_matrix()[:, self.current_vector_index])
            self.local_eigenvector_norm = float(eigenvector_norm)
            self.current_vector_index = self.current_vector_index + 1
            if self.use_smpc:
                self.set_compensator_flag()
            self.local_parameters[PcaLocalParameter.LOCAL_EIGENVECTOR_NORM] = self.local_eigenvector_norm


            self.log('ending eigenvector norms')

        except Exception as exception:
            self.log(f"\t{exception}\n")
            self.project_failed()

    def normalise_orthogonalised_matrix(self):
        try:
            # get the last eigenvector norm
            self.log('Normalising')
            self.all_global_eigenvector_norms.append(self.global_parameters[PcaGlobalParameter.GLOBAL_EIGENVECTOR_NORM])
            # divide all elements through the respective vector norm.
            for col in range(self.g_matrix.shape[1]):
                for row in range(self.g_matrix.shape[0]):
                    self.g_matrix[row, col] = self.g_matrix[row, col] / np.sqrt(self.all_global_eigenvector_norms[col])
            self.reset_eigenvector_norms()

            self.log('end normalising')
            # there is a next step!
            # if self.operation_status == OperationStatus.IN_PROGRESS:
            # self.operation_status = OperationStatus.DONE

        except Exception as exception:
            self.log(f"\t{exception}\n")
            self.project_failed()

    def projections(self):
        try:
            G = self.get_g_matrix()
            projections = np.dot(self.data, G)
            self.local_parameters[PcaLocalParameter.PROJECTIONS] = projections

        except Exception as exception:
            self.log(f"\t{exception}\n")
            self.project_failed()

    def create_local_result_obj(self):
        try:

            self.log('Creating result')
            if self.federated_qr:
                # result is available locally
                pd.DataFrame(self.get_g_matrix()).to_csv(
                    self.pca_dataset_file_path + '_' + str(self.project_id) + '.g')
            else:
                # result is computed on the server.
                Gi = self.global_parameters[PcaLocalParameter.GI_MATRIX]
                pd.DataFrame(Gi).to_csv(self.pca_dataset_file_path + '_' + str(self.project_id) + '.g')

            pd.DataFrame(self.get_row_names()).to_csv(
                self.pca_dataset_file_path + '_' + str(self.project_id) + '.row_names')
            pd.DataFrame(self.get_h_matrix()).to_csv(self.pca_dataset_file_path + '_' + str(self.project_id) + '.h')

            self.local_parameters[PcaLocalParameter.GI_MATRIX] = self.get_g_matrix()

        except Exception as exception:
            self.log(f"\t{exception}\n")
            self.project_failed()

    def get_g_matrix(self):
        return self.g_matrix

    def get_h_matrix(self):
        return self.h_matrix

    def reset_eigenvector_norms(self):
        self.all_global_eigenvector_norms = []
        self.current_vector_index = 0

    def get_local_vector_conorms(self):
        return self.local_vector_conorms





