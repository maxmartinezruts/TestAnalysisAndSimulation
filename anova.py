import pandas as pd
from statsmodels.stats.anova import AnovaRM


df1 = pd.read_csv("visual_anova_cm.csv")
print(df1)
aov1 = AnovaRM(df1, "magnitude", "subj_id", within=['motion',"vehicle"])

res1 = aov1.fit()
print(res1)
