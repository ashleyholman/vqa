import boto3

class S3Helper:
    BACKEND_BUCKET = 'vqa-ap-southeast-1'
    FRONTEND_BUCKET = 'vqa-web'

    def __init__(self):
        self.s3_client = boto3.client('s3')

    def list_objects(self, bucket_name=BACKEND_BUCKET, prefix=None):
        try:
            args = { 'Bucket': bucket_name }

            if prefix is not None:
                args['Prefix'] = prefix

            response = self.s3_client.list_objects_v2(**args)

            # return the list of object keys
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            print(f"Failed to list snapshots: {e}")
            return []

    def delete_prefix_recursive(self, bucket_name, prefix):
        # safety check to ensure prefix is not empty
        if not prefix:
            raise ValueError("Prefix is required")

        try:
            bucket = self.s3_resource.Bucket(bucket_name)

            for obj in bucket.objects.filter(Prefix=prefix):
                obj.delete()

            print(f"Successfully deleted all objects under prefix {prefix} from bucket {bucket_name}")
        except Exception as e:
            print(f"Failed to delete objects: {e}")

    def download_file(self, bucket_name, key, filename):
        try:
            self.s3_client.download_file(bucket_name, key, filename)
        except Exception as e:
            print(f"Failed to download file: {e}")
            raise e

    def upload_file(self, bucket_name, key, filename):
        try:
            self.s3_client.upload_file(filename, bucket_name, key)
        except Exception as e:
            print(f"Failed to upload file: {e}")
            raise e