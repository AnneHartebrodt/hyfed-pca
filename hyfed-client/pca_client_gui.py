"""
    Pca client GUI

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

from hyfed_client.widget.join_widget import JoinWidget
from hyfed_client.widget.hyfed_project_status_widget import HyFedProjectStatusWidget
from hyfed_client.util.hyfed_parameters import HyFedProjectParameter, ConnectionParameter, AuthenticationParameter

from pca_client.widget.pca_project_info_widget import PcaProjectInfoWidget
from pca_client.widget.pca_dataset_widget import PcaDatasetWidget
from pca_client.project.pca_client_project import PcaClientProject
from pca_client.util.pca_parameters import PcaProjectParameter

import threading

import logging
logger = logging.getLogger(__name__)


class PcaClientGUI:
    """ Pca Client GUI """

    def __init__(self):

        # create the join widget
        self.join_widget = JoinWidget(title="Pca Client",
                                      local_server_name="Localhost",
                                      external_server_name="Pca-Server",
                                      local_server_url="http://localhost:3850",
                                      external_server_url="https://federated.compbio.sdu.dk/api",
                                      local_compensator_name="Localhost",
                                      local_compensator_url="http://localhost:8001",
                                      external_compensator_name="TUM-Compensator",
                                      external_compensator_url="https://compensator.compbio.sdu.dk"
                                      )

        # show the join widget
        self.join_widget.show()

        # if join was NOT successful, terminate the client GUI
        if not self.join_widget.is_joined():
            return

        # if join was successful, get connection and authentication parameters from the join widget
        connection_parameters = self.join_widget.get_connection_parameters()
        authentication_parameters = self.join_widget.get_authentication_parameters()

        #  create Pca project info widget based on the authentication and connection parameters
        self.pca_project_info_widget = PcaProjectInfoWidget(title="Pca Project Info",
                                                                   connection_parameters=connection_parameters,
                                                                   authentication_parameters=authentication_parameters)

        # Obtain Pca project info from the server
        # the project info will be set in project_parameters attribute of the info widget
        self.pca_project_info_widget.obtain_project_info()

        # if Pca project info cannot be obtained from the server, exit the GUI
        if not self.pca_project_info_widget.project_parameters:
            return

        # add basic info of the project such as project id, project name, description, and etc to the info widget
        self.pca_project_info_widget.add_project_basic_info()

        # add Pca specific project info to the widget
        self.pca_project_info_widget.add_pca_project_info()

        # add accept and decline buttons to the widget
        self.pca_project_info_widget.add_accept_decline_buttons()

        # show project info widget
        self.pca_project_info_widget.show()

        # if participant declined to proceed, exit the GUI
        if not self.pca_project_info_widget.is_accepted():
            return

        # if user agreed to proceed, create and show the Pca dataset selection widget
        self.pca_dataset_widget = PcaDatasetWidget(title="Pca Dataset Selection")
        self.pca_dataset_widget.add_quit_run_buttons()
        self.pca_dataset_widget.show()

        # if the participant didn't click on 'Run' button, terminate the client GUI
        if not self.pca_dataset_widget.is_run_clicked():
            return

        # if participant clicked on 'Run', get all the parameters needed
        # to create the client project from the widgets
        connection_parameters = self.join_widget.get_connection_parameters()
        authentication_parameters = self.join_widget.get_authentication_parameters()
        project_parameters = self.pca_project_info_widget.get_project_parameters()

        server_url = connection_parameters[ConnectionParameter.SERVER_URL]
        compensator_url = connection_parameters[ConnectionParameter.COMPENSATOR_URL]
        username = authentication_parameters[AuthenticationParameter.USERNAME]
        token = authentication_parameters[AuthenticationParameter.TOKEN]
        project_id = authentication_parameters[AuthenticationParameter.PROJECT_ID]

        tool = project_parameters[HyFedProjectParameter.TOOL]
        algorithm = project_parameters[HyFedProjectParameter.ALGORITHM]
        project_name = project_parameters[HyFedProjectParameter.NAME]
        project_description = project_parameters[HyFedProjectParameter.DESCRIPTION]
        coordinator = project_parameters[HyFedProjectParameter.COORDINATOR]

        # PCA specific project info
        max_iterations = project_parameters[PcaProjectParameter.MAX_ITERATIONS]
        max_dimensions = project_parameters[PcaProjectParameter.MAX_DIMENSIONS]
        epsilon = project_parameters[PcaProjectParameter.EPSILON]
        center = project_parameters[PcaProjectParameter.CENTER]
        scale_variance = project_parameters[PcaProjectParameter.SCALE_VARIANCE]
        log2 = project_parameters[PcaProjectParameter.LOG2]
        send_final_result = project_parameters[PcaProjectParameter.SEND_FINAL_RESULT]
        federated_qr = project_parameters[PcaProjectParameter.FEDERATED_QR]
        speedup = project_parameters[PcaProjectParameter.SPEEDUP]
        use_smpc = project_parameters[PcaProjectParameter.USE_SMPC]

        pca_dataset_file_path = self.pca_dataset_widget.get_dataset_file_path()
        has_rownames = self.pca_dataset_widget.get_has_rownames()
        has_column_names = self.pca_dataset_widget.get_has_column_names()
        field_delimiter = self.pca_dataset_widget.get_field_delimiter()


        # create Pca client project
        pca_client_project = PcaClientProject(username=username,
                                                     token=token,
                                                    tool = tool,
                                                     server_url=server_url,
                                              compensator_url=compensator_url,
                                                     project_id=project_id,
                                                     algorithm=algorithm,
                                                     name=project_name,
                                                     description=project_description,
                                                     coordinator=coordinator,
                                                     result_dir='./pca_client/result',
                                                     log_dir='./pca_client/log',
                                              pca_datasets_file_path = pca_dataset_file_path,
                                              max_iterations=max_iterations,
                                              max_dimensions = max_dimensions,
                                              center = center,
                                              scale_variance=scale_variance,
                                              log2=log2,
                                              federated_qr=federated_qr,
                                              send_final_result = send_final_result,
                                              speedup = speedup,
                                              has_rownames = has_rownames,
                                              has_column_names=has_column_names,
                                              field_delimiter = field_delimiter,
                                              use_smpc = use_smpc
                                              )

        # run Pca client project as a thread
        pca_project_thread = threading.Thread(target=pca_client_project.run)
        pca_project_thread.setDaemon(True)
        pca_project_thread.start()

        # create and show Pca project status widget
        pca_project_status_widget = HyFedProjectStatusWidget(title="Pca Project Status",
                                                            project=pca_client_project)
        pca_project_status_widget.add_static_labels()
        pca_project_status_widget.add_progress_labels()
        pca_project_status_widget.add_status_labels()
        pca_project_status_widget.add_log_and_quit_buttons()
        pca_project_status_widget.show()


client_gui = PcaClientGUI()
