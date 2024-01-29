import tensorflow as tf
import tensorflow.compat.v1 as v1
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python.training import py_checkpoint_reader


def convert_tf1_to_tf2(file_name, output_prefix):
  """Converts a TF1 checkpoint to TF2.

  To load the converted checkpoint, you must build a dictionary that maps
  variable names to variable objects."""

  """Args:
    checkpoint_path: Path to the TF1 checkpoint.
    output_prefix: Path prefix to the converted checkpoint.

  Returns:
    Path to the converted checkpoint.
  """
  variables = {}
  reader = py_checkpoint_reader.NewCheckpointReader(file_name)#tf.train.load_checkpoint(checkpoint_path)
  dtypes = reader.get_variable_to_dtype_map()
  for key in dtypes.keys():
    variables[key] = tf.Variable(reader.get_tensor(key))
  return tf.train.Checkpoint(vars=variables).save(output_prefix)

# Prints all tensors and tensornames
#print_tensors_in_checkpoint_file('/cluster/home/emmalei/Master_project_network/ldgcnn/log/ldgcnn_model.ckpt', all_tensors=True, all_tensor_names=True, tensor_name="")

# Will hopefully make the TF1 checkpoint into a TF2 checkpoint
#converted_path = convert_tf1_to_tf2('/cluster/home/emmalei/Master_project_network/ldgcnn/log/ldgcnn_model.ckpt', '/cluster/home/emmalei/Master_project_network/ldgcnn/tf2_ldgcnn/converted-ldgcnn') 
#print("\n[Converted]") 
#print(converted_path)

# Restore checkpoint to be used
#ckpt = tf.train.Checkpoint(vars=variables) 
#ckpt.restore(converted_path).assert_consumed() 
#print("\nRestored")

# This is just for me to see each layer so I can recreate it in keras
reader = py_checkpoint_reader.NewCheckpointReader("/cluster/home/emmalei/Master_project_network/ldgcnn/log/ldgcnn_model.ckpt")
var_to_shape_map = reader.get_variable_to_shape_map()
i = 0
for key in var_to_shape_map:
  print("tensor_name: ", key)
  print(reader.get_tensor(key).shape) # Remove this is you want to print only variable names
  i+=1
print("Number of layers: "+str(i))
