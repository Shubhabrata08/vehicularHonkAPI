from django.db import models

# Create your models here.
from rest_framework import serializers

class ImageUploadSerializer(serializers.Serializer):
    numPatches = serializers.CharField()
    image = serializers.ImageField()
    def create(self, validated_data):
        # In this method, you can implement the logic to handle the uploaded image.
        # For example, you can save the image to your server or perform any necessary
        # processing on it.
        image_data = validated_data['image']
        # Your image processing logic goes here
        # Example: Saving the image to a specific location
        # image_data.name contains the original file name
        # image_data.file contains the file content
        
        # Return the processed image data or any other response you need
        return {'message': 'Image uploaded successfully'}

class TextSerializer(serializers.Serializer):
    numPatches = serializers.CharField()

