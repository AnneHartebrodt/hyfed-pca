"""
    Pca dataset widget to select the dataset file(s)

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

from hyfed_client.widget.hyfed_dataset_widget import HyFedDatasetWidget
from hyfed_client.util.gui import add_label_and_textbox, add_button, select_file_path, add_label_and_checkbox, radio_button
from tkinter import messagebox, StringVar, BooleanVar


import logging
logger = logging.getLogger(__name__)

class PcaDatasetWidget(HyFedDatasetWidget):
    """ This widget enables users to add the file/directory dialogs and select dataset files/directories """

    def __init__(self, title):

        super().__init__(title=title)

        self.dataset_file_path_entry = add_label_and_textbox(widget=self, label_text='Dataset File',
                                                             increment_row_number=False)

        self.dataset_file_path = ''  # initialized in set_dataset_file_path function
        self.has_rownames = False
        self.field_delimiter = '\t'

        add_button(widget=self, button_label="Browse", column_number=2, increment_row_number=True,
                   on_click_function=self.set_dataset_file_path_and_params)

        self.rownames = BooleanVar(self)
        add_label_and_checkbox(widget=self, label_text = 'Row names', variable=self.rownames, name='rownames')
        self.colnames = BooleanVar(self)
        add_label_and_checkbox(widget=self, label_text='Column names', variable = self.colnames, name='colnames')

        self.sep_var = StringVar(self)
        radio_button(widget=self, variable=self.sep_var, label_text = 'Field delimiter', label=",", value=',',
                     print_label_text = True, column=1)
        radio_button(widget=self, variable=self.sep_var, label_text='Field delimiter', label='tab', value='\t',
                     print_label_text=False, column = 2)
        radio_button(widget=self, variable=self.sep_var, label_text='Field delimiter', label=";", value=';',
                     print_label_text=False, increment_row_number=True, column=3)


    def get_has_rownames(self):
        return self.has_row_names

    def get_has_column_names(self):
        return self.has_column_names

    def get_field_delimiter(self):
        return self.field_delimiter

    def set_dataset_file_path_and_params(self):
        self.dataset_file_path = select_file_path(self.dataset_file_path_entry,
                                                  file_types=[('TSV files', '*.tsv'), ('CSV files', '*.csv'),
                                                              ('Txt files', '*.txt')])
    def get_dataset_file_path(self):
        return self.dataset_file_path


    def click_on_run(self):
        """ If participant clicked on Run, set run_clicked flag to True """


        self.has_row_names = self.rownames.get()
        self.has_column_names = self.colnames.get()
        self.field_delimiter = self.sep_var.get()

        if self.check_dataframe():

            self.run_clicked = True
            self.destroy()

    def check_files(self):
        if not self.dataset_file_path:
            messagebox.showerror("Dataset File Path", "Dataset file path cannot be empty!")
            return False

        if not self.check_dataframe():
            return False

        return True

    def check_dataframe(self):
        def isfloat(value):
            try:
                float(value)
                return True
            except ValueError:
                return False

        def line_is_double(line):
            conversion_possible = 0
            for l in range(len(line)):
                if isfloat(line[l]):
                    conversion_possible += 1
            return conversion_possible == len(line)

        try:
            with(open(self.dataset_file_path, 'r')) as handle:
                lines_to_read = 0
                # sanity check has been performed
                rownames = []
                for line in handle:
                    if lines_to_read >= 10:
                        break

                    line = line.split(self.field_delimiter)
                    # line only has 1 entry, probably wrong delim
                    if len(line) <= 1:
                        messagebox.showerror('Error', 'Could only detect one column.\
                                                        Are you sure your delimiter is correct?')
                        return False

                    if self.has_row_names:
                        rownames.append(line[0])
                        line = line[1:len(line)]

                    # User said first line is colnames but first line is float
                    if lines_to_read == 0:
                        if self.has_column_names and line_is_double(line):
                            messagebox.showerror('Error',
                                                 'All the values in your first line seem to be float (numeric). Are you sure you added column names?')
                            return False
                        elif not self.has_column_names and not line_is_double(line):
                            messagebox.showerror('Error',
                                                 'Your first line does not seem to be floats. Are you sure you do not\
                                                  have column names')
                            return False
                        else:
                            pass
                    else:
                        if not line_is_double(line):
                            messagebox.showerror('Error', 'Cannot convert line to numeric; line: ' + str(lines_to_read))
                            return False
                    lines_to_read = lines_to_read + 1

                # after the first 10 lines check if row names are numeric, if so, probably a mistake
                if line_is_double(rownames) and self.has_row_names:
                    messagebox.showerror('Error',
                                         'The first 10 lines of your file seem to have no row name (They are numeric). Are you sure you added (string) rownames?')
                    return False
        # catch all error message
        except Exception as exception:
            messagebox.showerror('Error', 'Your file is in a bad format')
            logger.debug(f"{exception}")
            return False

        return True

