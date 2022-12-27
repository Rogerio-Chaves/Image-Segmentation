from PIL import Image

import cv2
import numpy as np
import os
import tarfile
import tensorflow as tf

class DeepLabModel():
  ''' This class loads the deeplab model '''
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    ''' Create and load the pre-trained model.'''
    self.graph = tf.Graph()
    graph_def = None

    # Extract the frozen graph of the tar file.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
        break
    tar_file.close()

    if graph_def is None:
      #raise RuntimeError('Nao foi possivel encontrar o inference graph no arquivo tar.')
      raise RuntimeError('Cannot find the inference graph on the tar file.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.compat.v1.Session(graph=self.graph)

  def run(self, image, INPUT_TENSOR_NAME = 'ImageTensor:0', OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'):
    """Run the inference in one image.

    Parameters: 
    image: Objeto PIL.Image.
    INPUT_TENSOR_NAME: The name of the input tensor. default=ImageTensor
    OUTPUT_TENSOR_NAME: The name of the output tensor. default=SemanticPredictions

    Returns:
    resized_image: imagem de entrada RGB redimensionada  
    seg_map: Mapa de segmentação do `resized_image` 
    """
    width, height = image.size
    target_size = (2049,1025)  # tamanho das imagens of the Cityscapes dataset.
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

    batch_seg_map = self.sess.run(
      OUTPUT_TENSOR_NAME,
      feed_dict={INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        
    seg_map = batch_seg_map[0]  # espera o batch size = 1
    if len(seg_map.shape) == 2:
      seg_map = np.expand_dims(seg_map,-1)  # adiciona uma dimensão extra, necessária pro cv2.resize

    seg_map = cv2.resize(seg_map, (width,height), interpolation=cv2.INTER_NEAREST)
    return seg_map
