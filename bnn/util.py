#!/bin/python

import boto3
import os
import os.path
import zipfile
from urllib.request import urlretrieve
import pickle
import threading
import sys

# Get full path to a resource underneath this project (bayesian-neural-network-blogpost)
def full_path(name):
    base_dir_name = "bayesian-neural-network-blogpost"
    base_dir_list = os.getcwd().split("/")
    i = base_dir_list.index(base_dir_name)
    return "/".join(base_dir_list[0:i+1]) + "/" + name

# Save and fetch data and saved models from S3. Useful for working between AWS and local machine.

# TODO: change these to command line inputs.
bucket_name = 'kd-carnd'
key_name = 'bayesian-neural-network-blogpost/'
region_name = 'us-east-2'

def upload_s3(rel_path):
	bucket = boto3.resource('s3', region_name=region_name).Bucket(bucket_name)
	print("Uploading file", rel_path)
	bucket.upload_file(full_path(rel_path), key_name + rel_path, Callback=UploadProgressPercentage(rel_path))
	print("Finished uploading file", rel_path)

def download_s3(rel_path):
  bucket = boto3.resource('s3', region_name=region_name).Bucket(bucket_name)

  print("Downloading file", rel_path)
  try:
    bucket.download_file(key_name + rel_path, full_path(rel_path), Callback=DownloadProgressPercentage(rel_path))
    print("Finished downloading file", rel_path)
  except ClientError:
    print("Unable to find file", rel_path, ".")

class UploadProgressPercentage(object):
  def __init__(self, filename):
    self._filename = filename
    self._size = float(os.path.getsize(filename))
    self._seen_so_far = 0
    self._lock = threading.Lock()
  def __call__(self, bytes_amount):
    # To simplify we'll assume this is hooked up
    # to a single filename.
    with self._lock:
      self._seen_so_far += bytes_amount
      percentage = (self._seen_so_far / self._size) * 100
      sys.stdout.write(
        "\r%s  %s / %s  (%.2f%%)" % (
          self._filename, self._seen_so_far, self._size,
          percentage))
      sys.stdout.flush()

class DownloadProgressPercentage(object):
  def __init__(self, filename):
    self._filename = filename
    self._seen_so_far = 0
    self._lock = threading.Lock()
  def __call__(self, bytes_amount):
    # To simplify we'll assume this is hooked up
    # to a single filename.
    with self._lock:
      self._seen_so_far += bytes_amount
      sys.stdout.write(
        "\r%s --> %s bytes transferred" % (
          self._filename, self._seen_so_far))
      sys.stdout.flush()

def download_file(url, file):
  """
  Download file from <url>
  :param url: URL to file
  :param file: Local file path
  """
  if os.path.isfile(file) == False:
    print("Unable to find " + file + ". Downloading now...")
    urlretrieve(url, file)
    print('Download Finished!')
  else:
    print(file + " already downloaded.")

def unzip_data(zip_file_name, location):
  """
  unzip file 
  :param zip_file_name: name of zip file
  :param location: path to unzip location
  """
  if os.path.isfile(full_path(zip_file_name)) == False:
    with zipfile.ZipFile(full_path(zip_file_name), "r") as zip_ref:
      print("Extracting zipfile " + zip_file_name + "...")
      zip_ref.extractall(full_path(location))
  else:
    print("Zipfile", zip_file_name, "already unzipped.")

def zipdir(path, ziph):
  # ziph is zipfile handle
  for root, dirs, files in os.walk(path):
    for file in files:
      ziph.write(os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))

def open_pickle_file(file_name):
  """
  open a pickled file
  :param file_name: name of file
  """
  print("Unpickling file " + file_name)
  full_file_name = full_path(file_name)
  with open(full_file_name, mode='rb') as f:
    return pickle.load(f)

def save_pickle_file(file, data):
  """
  save an object as a pickled file
  :param file: name of file
  :param data: python object to save
  """
  abs_file = full_path(file)
  pickle.dump(data, open(abs_file, "wb" ) )

def stop_instance():
  ec2 = boto3.resource('ec2', region_name='us-east-1')
  instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])
  ids = [i.id for i in instances]
  ec2.instances.filter(InstanceIds=ids).stop() # .terminate()

def isAWS():
  cwd_path = os.getcwd().split("/")
  if len(cwd_path) > 2 and cwd_path[2] == 'kyledorman':
    return False
  return True

class BatchConfig(object):
  def __init__(self, encoder, dataset):
    self.encoder = encoder
    self.dataset = dataset

  def batch_folder(self):
    return "batch_data/{}_{}".format(self.encoder, self.dataset)

  def info(self):
    print("encoder:", self.encoder)
    print("dataset:", self.dataset)

class BayesianConfig(object):
  def __init__(self, encoder, dataset, batch_size, epochs, monte_carlo_simulations):
    self.encoder = encoder
    self.dataset = dataset
    self.epochs = epochs
    self.batch_size = batch_size
    self.monte_carlo_simulations = monte_carlo_simulations

  def info(self):
    print("encoder:", self.encoder)
    print("batch_size:", self.batch_size)
    print("epochs:", self.epochs)
    print("dataset:", self.dataset)
    print("monte_carlo_simulations:", self.monte_carlo_simulations)

  def model_file(self):
    return "model_{}_{}_{}_{}_{}.ckpt".format(self.encoder, self.dataset, self.batch_size, self.epochs, self.monte_carlo_simulations)

  def csv_log_file(self):
    return "model_training_logs_{}_{}_{}_{}_{}.csv".format(self.encoder, self.dataset, self.batch_size, self.epochs, self.monte_carlo_simulations)
