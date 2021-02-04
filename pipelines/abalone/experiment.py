

import time, os, sys
import sagemaker, boto3
import numpy as np
import pandas as pd
import itertools
from pprint import pprint

sess = boto3.Session()
sm   = sess.client('sagemaker')
role = sagemaker.get_execution_role()
sagemaker_session = sagemaker.Session(boto_session=sess)

bucket_name = sagemaker_session.default_bucket()

from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent
from smexperiments.tracker import Tracker

training_experiment = Experiment.create(
                                experiment_name = f"cifar10-training-experiment-{int(time.time())}",
                                description     = "Hypothesis: Custom model architecture delivers higher validation accuracy for classification compared to ResNet50 and VGG on the CIFAR10 dataset",
                                sagemaker_boto_client=sm)
