"""
    Pca project serializer to serialize project specific fields

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

from rest_framework import serializers
from hyfed_server.serializer.hyfed_serializers import HyFedProjectSerializer


class PcaProjectSerializer(HyFedProjectSerializer):
    """ Serializes the Pca project model to serve a WebApp/client request """

    max_iterations = serializers.SerializerMethodField()
    max_dimensions = serializers.SerializerMethodField()
    center = serializers.SerializerMethodField()
    scale_variance = serializers.SerializerMethodField()
    log2 = serializers.SerializerMethodField()
    federated_qr = serializers.SerializerMethodField()
    send_final_result = serializers.SerializerMethodField()
    current_iteration = serializers.SerializerMethodField()
    epsilon = serializers.SerializerMethodField()
    speedup = serializers.SerializerMethodField()

    def get_max_iterations(self, instance):
        return instance.max_iterations

    def get_max_dimensions(self, instance):
        return instance.max_dimensions

    def get_center(self, instance):
        return instance.center

    def get_scale_variance(self, instance):
        return instance.scale_variance

    def get_log2(self, instance):
        return instance.log2

    def get_federated_qr(self, instance):
        return  instance.federated_qr

    def get_send_final_result(self, instance):
        return instance.send_final_result

    def get_current_iteration(self, instance):
        return instance.current_iteration

    def get_epsilon(self, instance):
        return instance.epsilon

    def get_speedup(self, instance):
        return instance.speedup

    class Meta(HyFedProjectSerializer.Meta):
        fields = HyFedProjectSerializer.Meta.fields + ('max_iterations', 'max_dimensions', 'center',
                                                       'scale_variance', 'log2', 'federated_qr', 'send_final_result',
                                                       'current_iteration', 'epsilon','speedup',)
