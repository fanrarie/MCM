import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import exe5model

if __name__ == "__main__":
    # 加载数据
    df1 = pd.read_excel('附件1.xlsx')
    df2_predicate = pd.read_excel('xxr附件2预测出院时间.xlsx')
    df3 = pd.read_excel('附件3.xlsx', sheet_name='Sheet1')
    df_proportion = pd.read_excel('问题5比例床位数.xlsx', sheet_name='Sheet1')
    flag,pre_op_stay_time,pre_op_wait_time,admission_wait_time=exe5model.problem2_model(df3, df1, df_proportion, df2_predicate)