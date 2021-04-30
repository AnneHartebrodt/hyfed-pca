"""
    A widget to add Pca specific project info

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

from hyfed_client.widget.hyfed_project_info_widget import HyFedProjectInfoWidget
from hyfed_client.util.gui import add_label_and_textbox
from hyfed_client.util.hyfed_parameters import HyFedProjectParameter
from stats_client.util.stats_parameters import StatsProjectParameter
from pca_client.util.pca_parameters import PcaProjectParameter
from pca_client.util.pca_algorithms import PcaAlgorithm


class PcaProjectInfoWidget(HyFedProjectInfoWidget):
    def __init__(self, title, connection_parameters, authentication_parameters):

        super().__init__(title=title, connection_parameters=connection_parameters,
                         authentication_parameters=authentication_parameters)

    # Stats project specific info
    def add_stats_project_info(self):
        add_label_and_textbox(self, label_text="Dimensions",
                              value=self.project_parameters[PcaProjectParameter.MAX_DIMENSIONS], status='disabled')

        add_label_and_textbox(self, label_text="Epsilon",
                              value=self.project_parameters[PcaProjectParameter.EPSILON], status='disabled')

        add_label_and_textbox(self, label_text="Max iterations",
                              value=self.project_parameters[PcaProjectParameter.MAX_ITERATIONS],
                              status='disabled')
        add_label_and_textbox(self, label_text = "Center",
                              value = self.project_parameters[PcaProjectParameter.CENTER],
                              status = 'disabled')
        add_label_and_textbox(self, label_text = 'Scale variance',
                              value = self.project_parameters[PcaProjectParameter.SCALE_VARIANCE],
                              status = 'disabled')
        add_label_and_textbox(self, label_text = 'Log2 transform',
                              value = self.project_parameters[PcaProjectParameter.LOG2],
                              status = 'disabled')
        add_label_and_textbox(self, label_test = 'Federated QR',
                              value = self.project_parameters[PcaProjectParameter.FEDERATED_QR],
                              status = 'disabled')
        add_label_and_textbox(self, label_test='Speedup',
                              value=self.project_parameters[PcaProjectParameter.SPEEDUP],
                              status='disabled')

    # Pca project specific info
    def add_pca_project_info(self):
        pass
