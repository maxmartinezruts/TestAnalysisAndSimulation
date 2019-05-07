import pandas as pd
from statsmodels.stats.anova import AnovaRM


df1 = pd.read_csv("visual_kv.csv")
aov1 = AnovaRM(df1, "rmsudot", "subj_id", within=["motion", "vehicle"])

res1 = aov1.fit()
print(res1)
