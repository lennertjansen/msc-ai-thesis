import numpy as np
import pandas as pd

array = [1, 2, 3, 4, 5, 6]
average = np.mean(array)
df = pd.DataFrame(array)
print(array)
print(df - average)