# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
# Reading the CSV
file_path = r'C:\Users\dbal\anaconda_projects\PotentialTalentsNLP\ExtendedPotentialTalents.csv'
full_df = pd.read_csv(file_path)
full_df.head()