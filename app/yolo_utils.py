import time
from pathlib import Path
from detect import run
import boto3
from botocore import exceptions as boto_exceptions
from loguru import logger
import yaml
import os
from bson import ObjectId

IMAGES_BUCKET = os.environ['BUCKET_NAME']
IMAGES_PREFIX = os.environ['BUCKET_PREFIX']
TABLE_NAME    = os.environ['TABLE_NAME']

aws_profile = os.getenv("AWS_PROFILE", None)
if aws_profile is not None and aws_profile == "dev":
    boto3.setup_default_session(profile_name=aws_profile)

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

def convert_objectid(data):
    if isinstance(data, dict):
        return {key: convert_objectid(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_objectid(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)  # Convert ObjectId to string
    else:
        return data

# Convert the labels array to the DynamoDB format
def convert_labels(labels):
    return [
        {
            "M": {
                "class": {"S": label["class"]},
                "cx": {"N": str(label["cx"])},
                "cy": {"N": str(label["cy"])},
                "width": {"N": str(label["width"])},
                "height": {"N": str(label["height"])}
            }
        } for label in labels
    ]

def upload_image_to_s3(bucket_name, key, image_path):
    try:
        s3_client = boto3.client('s3')
    except boto_exceptions.ProfileNotFound as e:
        logger.exception(f"Upload to {bucket_name}/{key} failed. A ProfileNotFound has occurred.\n{str(e)}")
        return f"Upload to {bucket_name}/{key} failed. A ProfileNotFound has occurred.\n{str(e)}", 500
    except boto_exceptions.EndpointConnectionError as e:
        logger.exception(f"Upload to {bucket_name}/{key} failed. An EndpointConnectionError has occurred.\n{str(e)}")
        return f"Upload to {bucket_name}/{key} failed. An EndpointConnectionError has occurred.\n{str(e)}", 500
    except boto_exceptions.NoCredentialsError as e:
        logger.exception(f"Upload to {bucket_name}/{key} failed. A NoCredentialsError has occurred.\n{str(e)}")
        return f"Upload to {bucket_name}/{key} failed. A NoCredentialsError has occurred.\n{str(e)}", 500
    except boto_exceptions.ClientError as e:
        logger.exception(f"Upload to {bucket_name}/{key} failed. A ClientError has occurred.\n{str(e)}")
        return f"Upload to {bucket_name}/{key} failed. A ClientError has occurred.\n{str(e)}", 500
    except Exception as e:
        logger.exception(f"Upload to {bucket_name}/{key} failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}")
        return f"Upload to {bucket_name}/{key} failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}", 500

    try:
        with image_path.open('rb') as img:
            s3_client.put_object(Bucket=bucket_name, Key=key, Body=img)
    except FileNotFoundError as e:
        logger.exception(f"Upload to {bucket_name}/{key} failed. A FileNotFoundError has occurred.\n{str(e)}")
        return f"Upload to {bucket_name}/{key} failed. A FileNotFoundError has occurred.\n{str(e)}", 500
    except PermissionError as e:
        logger.exception(f"Upload to {bucket_name}/{key} failed. A PermissionError has occurred.\n{str(e)}")
        return f"Upload to {bucket_name}/{key} failed. A PermissionError has occurred.\n{str(e)}", 500
    except IsADirectoryError as e:
        logger.exception(f"Upload to {bucket_name}/{key} failed. An IsADirectoryError has occurred.\n{str(e)}")
        return f"Upload to {bucket_name}/{key} failed. An IsADirectoryError has occurred.\n{str(e)}", 500
    except OSError as e:
        logger.exception(f"Upload to {bucket_name}/{key} failed. An {type(e).__name__} has occurred.\n{str(e)}")
        return f"Upload to {bucket_name}/{key} failed. An {type(e).__name__} has occurred.\n{str(e)}", 500
    except boto_exceptions.ParamValidationError as e:
        logger.exception(f"Upload to {bucket_name}/{key} failed. A ParamValidationError has occurred.\n{str(e)}")
        return f"Upload to {bucket_name}/{key} failed. A ParamValidationError has occurred.\n{str(e)}", 500
    except boto_exceptions.ClientError as e:
        logger.exception(f"Upload to {bucket_name}/{key} failed. A ClientError has occurred.\n{str(e)}")
        return f"Upload to {bucket_name}/{key} failed. A ClientError has occurred.\n{str(e)}", 500
    except Exception as e:
        logger.exception(f"Upload to {bucket_name}/{key} failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}")
        return f"Upload to {bucket_name}/{key} failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}", 500

    logger.info(f"Upload to {bucket_name}/{key} succeeded.")
    return f"Upload to {bucket_name}/{key} succeeded.", 200

def download_image_from_s3(bucket_name, key, image_path):
    if not os.path.exists(IMAGES_PREFIX):
        os.makedirs(IMAGES_PREFIX)

    try:
        s3_client = boto3.client('s3')
    except boto_exceptions.ProfileNotFound as e:
        logger.exception(f"Download from {bucket_name}/{key} failed. A ProfileNotFound has occurred.\n{str(e)}")
        return f"Download from {bucket_name}/{key} failed. A ProfileNotFound has occurred.\n{str(e)}", 500
    except boto_exceptions.EndpointConnectionError as e:
        logger.exception(f"Download from {bucket_name}/{key} failed. An EndpointConnectionError has occurred.\n{str(e)}")
        return f"Download from {bucket_name}/{key} failed. An EndpointConnectionError has occurred.\n{str(e)}", 500
    except boto_exceptions.NoCredentialsError as e:
        logger.exception(f"Download from {bucket_name}/{key} failed. A NoCredentialsError has occurred.\n{str(e)}")
        return f"Download from {bucket_name}/{key} failed. A NoCredentialsError has occurred.\n{str(e)}", 500
    except boto_exceptions.ClientError as e:
        logger.exception(f"Download from {bucket_name}/{key} failed. A ClientError has occurred.\n{str(e)}")
        return f"Download from {bucket_name}/{key} failed. A ClientError has occurred.\n{str(e)}", 500
    except Exception as e:
        logger.exception(f"Download from {bucket_name}/{key} failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}")
        return f"Download from {bucket_name}/{key} failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}", 500

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
    except boto_exceptions.ClientError as e:
        logger.exception(f"Download from {bucket_name}/{key} failed. A ClientError has occurred.\n{str(e)}")
        return f"Download from {bucket_name}/{key} failed. A ClientError has occurred.\n{str(e)}", 500
    except Exception as e:
        logger.exception(f"Download from {bucket_name}/{key} failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}")
        return f"Download from {bucket_name}/{key} failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}", 500

    try:
        with open(image_path, 'wb') as img:
            img.write(response['Body'].read())
    except PermissionError as e:
        logger.exception(f"Download from {bucket_name}/{key} failed. A PermissionError has occurred.\n{str(e)}")
        return f"Download from {bucket_name}/{key} failed. A PermissionError has occurred.\n{str(e)}", 500
    except IsADirectoryError as e:
        logger.exception(f"Download from {bucket_name}/{key} failed. An IsADirectoryError has occurred.\n{str(e)}")
        return f"Download from {bucket_name}/{key} failed. An IsADirectoryError has occurred.\n{str(e)}", 500
    except OSError as e:
        logger.exception(f"Download from {bucket_name}/{key} failed. An {type(e).__name__} has occurred.\n{str(e)}")
        return f"Download from {bucket_name}/{key} failed. An {type(e).__name__} has occurred.\n{str(e)}", 500
    except Exception as e:
        logger.exception(f"Download from {bucket_name}/{key} failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}")
        return f"Download from {bucket_name}/{key} failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}", 500

    logger.info(f"Download from {bucket_name}/{key} to {image_path} succeeded.")
    return f"Download from {bucket_name}/{key} to {image_path} succeeded.", 200

def identify(img_name, prediction_id):
    logger.info(f'Prediction: {prediction_id}. Start processing')

    if not img_name:
        logger.exception(f'Prediction: {prediction_id}/{img_name}. No image parameter was passed or is empty.')
        return f'Prediction: {prediction_id}/{img_name}. No image parameter was passed or is empty.', 500

    # The bucket name and prefix are provided as env variables BUCKET_NAME and BUCKET_PREFIX respectively.
    original_img_path = IMAGES_PREFIX + "/" + img_name

    # Download img_name from S3 and store it to a local path from the original_img_path variable.
    response = download_image_from_s3(IMAGES_BUCKET, original_img_path, original_img_path)

    if int(response[1]) != 200:
        logger.exception(f'Prediction: {prediction_id}/{img_name} failed.')
        return f'Prediction: {prediction_id}/{img_name} failed.\n\n{response[0]}', response[1]

    # Predicts the objects in the image
    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=original_img_path,
        project='static/data',
        name=prediction_id,
        save_txt=True
    )

    logger.info(f'Prediction: {prediction_id}/{img_name}. Done')

    # Parse prediction labels and create a summary
    pred_summary_path = Path(f'static/data/{prediction_id}/labels/{img_name.split(".")[0]}.txt')
    if not pred_summary_path.exists():
        logger.exception(f'Prediction: {prediction_id}/{img_name} failed. Prediction result not found')
        return f'Prediction: {prediction_id}/{img_name} failed. Prediction result not found', 404

    with pred_summary_path.open(mode='r', encoding='utf-8') as f:
        labels = f.read().splitlines()
        labels = [line.split(' ') for line in labels]
        labels = [{
            'class': names[int(l[0])],
            'cx': float(l[1]),
            'cy': float(l[2]),
            'width': float(l[3]),
            'height': float(l[4]),
        } for l in labels]

    # This is the path for the predicted image with labels
    # The predicted image typically includes bounding boxes drawn around the detected objects, along with class labels and possibly confidence scores.
    predicted_img_path = Path(f'static/data/{prediction_id}/{img_name}')

    # Uploads the predicted image (predicted_img_path) to S3.
    response = upload_image_to_s3(IMAGES_BUCKET, original_img_path, predicted_img_path)

    if int(response[1]) != 200:
        logger.exception(f'Prediction: {prediction_id}/{img_name} failed.')
        return f'Prediction: {prediction_id}/{img_name} failed.\n\n{response[0]}', response[1]

    prediction_summary = {
        'prediction_id': prediction_id,
        'original_img_path': original_img_path,
        'predicted_img_path': predicted_img_path.as_posix(),
        'labels': labels,
        'time': time.time()
    }

    logger.info(f'Prediction: {prediction_id}/{img_name} was successful.\n\nPrediction summary:\n{prediction_summary}')
    return prediction_summary, 200

def write_to_db(chat_id, prediction_summary):
    try:
        dynamodb_client = boto3.client('dynamodb')
    except boto_exceptions.ProfileNotFound as e:
        logger.exception(f"Writing to dynamodb failed. A ProfileNotFound has occurred.\n{str(e)}")
        return f"Writing to dynamodb failed. A ProfileNotFound has occurred.\n{str(e)}", 500
    except boto_exceptions.ClientError as e:
        logger.exception(f"Writing to dynamodb failed. A ClientError has occurred.\n{str(e)}")
        return f"Writing to dynamodb failed. A ClientError has occurred.\n{str(e)}", 500
    except boto_exceptions.EndpointConnectionError as e:
        logger.exception(f"Writing to dynamodb failed. An EndpointConnectionError has occurred.\n{str(e)}")
        return f"Writing to dynamodb failed. An EndpointConnectionError has occurred.\n{str(e)}", 500
    except boto_exceptions.NoCredentialsError as e:
        logger.exception(f"Writing to dynamodb failed. A NoCredentialsError has occurred.\n{str(e)}")
        return f"Writing to dynamodb failed. A NoCredentialsError has occurred.\n{str(e)}", 500
    except Exception as e:
        logger.exception(f"Writing to dynamodb failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}")
        return f"Writing to dynamodb failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}", 500

    # Insert data into the table
    try:
        data = {
            'predictionId': {'S': str(prediction_summary['prediction_id'])},
            'chatId': {'N': str(chat_id)},
            'originalImgPath': {'S': prediction_summary['original_img_path']},
            'predictedImgPath': {'S': prediction_summary['predicted_img_path']},
            'labels': {'L': convert_labels(prediction_summary['labels'])},
            'time': {'N': str(prediction_summary['time'])}
        }

        response = dynamodb_client.put_item(
            TableName=TABLE_NAME,
            Item=data,
            ReturnConsumedCapacity='TOTAL',
            ReturnItemCollectionMetrics='SIZE',
            ReturnValues='ALL_OLD'
        )
    except dynamodb_client.exceptions.ConditionalCheckFailedException as e:
        logger.exception(f"Writing to dynamodb failed. A ConditionalCheckFailedException has occurred.\n{str(e)}")
        return f"Writing to dynamodb failed. A ConditionalCheckFailedException has occurred.\n{str(e)}", 500
    except dynamodb_client.exceptions.ProvisionedThroughputExceededException as e:
        logger.exception(f"Writing to dynamodb failed. A ProvisionedThroughputExceededException has occurred.\n{str(e)}")
        return f"Writing to dynamodb failed. A ProvisionedThroughputExceededException has occurred.\n{str(e)}", 500
    except dynamodb_client.exceptions.ResourceNotFoundException as e:
        logger.exception(f"Writing to dynamodb failed. A ResourceNotFoundException has occurred.\n{str(e)}")
        return f"Writing to dynamodb failed. A ResourceNotFoundException has occurred.\n{str(e)}", 500
    except dynamodb_client.exceptions.ItemCollectionSizeLimitExceededException as e:
        logger.exception(f"Writing to dynamodb failed. A ItemCollectionSizeLimitExceededException has occurred.\n{str(e)}")
        return f"Writing to dynamodb failed. A ItemCollectionSizeLimitExceededException has occurred.\n{str(e)}", 500
    except dynamodb_client.exceptions.TransactionConflictException as e:
        logger.exception(f"Writing to dynamodb failed. A TransactionConflictException has occurred.\n{str(e)}")
        return f"Writing to dynamodb failed. A TransactionConflictException has occurred.\n{str(e)}", 500
    except dynamodb_client.exceptions.RequestLimitExceeded as e:
        logger.exception(f"Writing to dynamodb failed. A RequestLimitExceeded has occurred.\n{str(e)}")
        return f"Writing to dynamodb failed. A RequestLimitExceeded has occurred.\n{str(e)}", 500
    except dynamodb_client.exceptions.InternalServerError as e:
        logger.exception(f"Writing to dynamodb failed. A InternalServerError has occurred.\n{str(e)}")
        return f"Writing to dynamodb failed. A InternalServerError has occurred.\n{str(e)}", 500
    except boto_exceptions.ClientError as e:
        logger.exception(f"Writing to dynamodb failed. A ClientError has occurred.\n{str(e)}")
        return f"Writing to dynamodb failed. A ClientError has occurred.\n{str(e)}", 500
    except Exception as e:
        logger.exception(f"Writing to dynamodb failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}")
        return f"Writing to dynamodb failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}", 500

    logger.info("Image prediction details were written to the DB successfully")
    logger.info(response)

    converted_data = convert_objectid(prediction_summary)
    return converted_data, 200

def send_to_sqs(queue_name, message_body):
    try:
        sqs_client = boto3.client('sqs')
    except boto_exceptions.ProfileNotFound as e:
        logger.exception(f"Sending message to SQS failed. A ProfileNotFound has occurred.\n{str(e)}")
        return f"Sending message to SQS failed. A ProfileNotFound has occurred.\n{str(e)}", 500
    except boto_exceptions.EndpointConnectionError as e:
        logger.exception(f"Sending message to SQS failed. An EndpointConnectionError has occurred.\n{str(e)}")
        return f"Sending message to SQS failed. An EndpointConnectionError has occurred.\n{str(e)}", 500
    except boto_exceptions.NoCredentialsError as e:
        logger.exception(f"Sending message to SQS failed. A NoCredentialsError has occurred.\n{str(e)}")
        return f"Sending message to SQS failed. A NoCredentialsError has occurred.\n{str(e)}", 500
    except boto_exceptions.ClientError as e:
        logger.exception(f"Sending message to SQS failed. A ClientError has occurred.\n{str(e)}")
        return f"Sending message to SQS failed. A ClientError has occurred.\n{str(e)}", 500
    except Exception as e:
        logger.exception(f"Sending message to SQS failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}")
        return f"Sending message to SQS failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}", 500

    try:
        response = sqs_client.send_message(
            QueueUrl=queue_name,
            MessageBody=message_body
        )
    except boto_exceptions.ParamValidationError as e:
        logger.exception(f"Sending message to SQS failed. A ParamValidationError has occurred.\n{str(e)}")
        return f"Sending message to SQS failed. A ParamValidationError has occurred.\n{str(e)}", 500
    except boto_exceptions.ClientError as e:
        logger.exception(f"Sending message to SQS failed. A ClientError has occurred.\n{str(e)}")
        return f"Sending message to SQS failed. A ClientError has occurred.\n{str(e)}", 500
    except Exception as e:
        logger.exception(f"Sending message to SQS failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}")
        return f"Sending message to SQS failed. An Unknown {type(e).__name__} has occurred.\n{str(e)}", 500

    logger.info(f"Message sent successfully. Message ID: {response['MessageId']}")
    return f"Message sent successfully. Message ID: {response['MessageId']}", 200