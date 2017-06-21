#!/bin/python

from util import open_pickle_file, download_file, unzip_data

def get_traffic_sign_data():
  url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip"
  zip_file = "traffic-sign-data.zip"

  download_file(url, zip_file)
  unzip_data(zip_file, "data/traffic-sign")

  train = open_pickle_file("data/traffic-sign/train.p")
  test = open_pickle_file("data/traffic-sign/test.p")
  valid = open_pickle_file("data/traffic-sign/valid.p")

  return ((train['features'], train['labels']), (test['features'], test['labels']), (valid['features'], valid['labels']))
