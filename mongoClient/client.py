import pymongo
import json
import csv
import numpy as np
import pandas as pd

class ssumtagoClient:
    def __init__(self,url):
        self.url=url
