from rest_framework import serializers
from apps.endpoints.models import Endpoint
from apps.endpoints.models import MLAlgorithm
from apps.endpoints.models import MLAlgorithmStatus
from apps.endpoints.models import MLRequest

class EndpointSerializer(serializers.ModelSerializer):
    class Meta:
        model = Endpoint
        read_only_fields = ('id','name','owner','created_at')
        fields = read_only_fields

class MLAlgorithmSerializer(serializers.ModelSerializer):
    
    current_status =serializers.SerializerMethodField(read_only=True)

    def get_current_status(self, mlalgorithm):
        result = MLAlgorithmStatus.object.filter(parent_mlalgorithm=mlalgorithm).lastest('created_at').status
        return result 

    class Meta:
        model = MLAlgorithmStatus
        read_only_fields = ('id','active')
        fields = (
            'id',
            'active',
            'status',
            'created_by',
            'created_at',
            'parent_mlalgorithm'
        )

class MLAlgorithmStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLAlgorithmStatus
        read_only_fields = ("id", "active")
        fields = ("id", "active", "status", "created_by", "created_at",
                            "parent_mlalgorithm")
                            

class MLRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLRequest
        read_only_fields = (
            'id',
            'input_data',
            'full_response',
            'response',
            'created_at',
            'parent_mlalgorithm',
        )
        fields = (
            'id',
            'input_data',
            'response',
            'feedback',
            'created_at',
            'parent_mlalgorithm'
        )
        
