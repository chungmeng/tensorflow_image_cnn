#### Personal instructions

Let's say you have a directory of images: ```/data/photos/class1```, ```/data/photos/class2```,...

For this step make sure there are only RGB images in your dataset.

To create TFRecord from directory of images in /data/tf_record:
```
python3 convert_data.py --dataset_name=standard --dataset_dir=/data/
```

To train model from scratch using TFRecords:

```
python3 train_image_classifier.py --train_dir=/tmp/train_logs \
  --dataset_name=standard \
  --dataset_split_name=train \
  --dataset_dir=/data/tf_record \
  --model_name=inception_v3
```

## Evaluating model
You can evaluate the model using:
```
python3 eval_image_classifier.py \
  --alsologtoostderr \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --dataset_dir=${DATASET_DIR} \
  --dataset_name=standard \
  --dataset_split=validation \
  --model_name=inception_v3
```
This will show you the accuracy of the model as well as confusion matrix and
also a whole list of wrongly classified images.

## Export and freeze graph for inference

If you then want to use the resulting model with your own or pretrained
checkpoints as part of a mobile model, you can run export and freeze the graph
def with the variables inlined as constants using:

Export graph:

```
python3 export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --output_file=/tmp/inception_v3_inf_graph.pb \
  --dataset_name=standard \
  --dataset_dir=/path/to/tf_record
```

*Note: This should be run from tensorflow main repository*

```shell
bazel build tensorflow/python/tools:freeze_graph

bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=/tmp/inception_v3_inf_graph.pb \
  --input_checkpoint=/checkpoints/model.ckpt-10000 \
  --input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
  --output_node_names=InceptionV3/Predictions/Reshape_1
```

## Creating Saved Model
```
python3 export_saved_model.py \
  --input_layer=input \
  --output_layer=InceptionV3/Predictions/Reshape_1 \
  --graph_file=/tmp/frozen_inception_v3.pb \
  --output_dir=/path/to/saved_model
```

## Deploying saved model using tensorflow serving

To deploy the saved model you can use the tensorflow-serving dockerfile and
modify it to copy your saved model bundle into the container if you want to deploy on deathbox.




