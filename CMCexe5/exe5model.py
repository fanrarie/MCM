import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np


##########################################
def calculate_bed_availability(df2_predicate):
    # 病床可用性数据
    # 计算每天出院的人数
    df2_predicate_discharge_counts = df2_predicate['出院时间'].value_counts().sort_index()

    # 创建bed_availability_df DataFrame的日期范围
    # 注意：这里我们使用.date()来确保min()和max()返回的是日期对象，而不是datetime对象
    start_date = df2_predicate['出院时间'].min().date()
    end_date = df2_predicate['出院时间'].max().date()
    date_range = pd.date_range(start=start_date, end=end_date)
    date_range = pd.to_datetime(date_range)
    # 初始化bed_availability_df DataFrame
    # 这里我们假设初始时没有额外的病床释放，所以所有值都设为0
    bed_availability_df = pd.DataFrame({
        '日期': date_range,
        '床位数': 0  # 更改为'病床释放数'以更准确地描述这一列
    })

    # 使用iterrows()迭代bed_availability_df的行
    for index, row in bed_availability_df.iterrows():
        row_date = row['日期']  # 这里不需要将日期转换为datetime，因为它已经是date_range的一部分
        if row_date in df2_predicate_discharge_counts.index:
            # 直接从df2_predicate_discharge_counts中获取出院人数
            # 注意：这里假设df2_predicate_discharge_counts的索引与bed_availability_df的'日期'列相匹配
            bed_availability_df.at[index, '床位数'] = df2_predicate_discharge_counts[row_date]

    # 查看更新后的bed_availability_df
    return bed_availability_df

##############################################
def ratio_suitable(df_proportion, df2_predicate, target_date_str):
    # 设定目标日期
    target_date = pd.to_datetime(target_date_str)

    # 筛选在目标日期仍住院的病人
    hospitalized_patients = df2_predicate[
        (df2_predicate['入院时间'] <= target_date) & (df2_predicate['出院时间'] > target_date)]

    # 统计各类型病人在目标日期的病床占有数
    bed_occupancy = hospitalized_patients.groupby('类型')['序号'].count().reset_index(name='病人数')

    # 比较是否超过
    for index, row in df_proportion.iterrows():
        row_type = row['类型']
        # 检查row_type是否存在于bed_occupancy的'类型'列中
        if row_type in bed_occupancy['类型'].values:
            # 使用布尔索引找到对应的病人数
            patient_count = bed_occupancy.loc[bed_occupancy['类型'] == row_type, '病人数'].values[0]
            # 比较病床数和病人数
            if row['病床数'] < patient_count:
                print(f'{target_date}各类型病人数:')
                print(bed_occupancy)
                print(f'{target_date}各类型分配的比例病床数:')
                print(df_proportion)
                return False
                # 如果row_type不存在于bed_occupancy中，则不需要特别处理，因为我们已经检查了所有存在的类型

    return True

def ratio_suitable_table2(df_proportion, df2_predicate):
    start_day=df2_predicate['入院时间'].min()
    end_day=df2_predicate['出院时间'].max()
    # 生成日期范围（注意：这里我们假设我们只关心到最晚出院日期当天的数据）
    # 你可以根据需要调整 freq 参数，比如 'D' 表示每天，'B' 表示工作日等
    date_range = pd.date_range(start=start_day, end=end_day, freq='D')

    # 遍历日期范围
    for target_date in date_range:
        flag = ratio_suitable(df_proportion, df2_predicate, target_date)
        if not flag:  # 如果 flag 是 False，则直接返回
            return False

            # 如果所有日期都通过了检查，则返回 True
    return True
#############################################
# 计算术后观察时间
def calculate_observation_time(row):
    if row['类型'] == '白内障(双眼)':
        if pd.notna(row['第二次手术时间']):
            return (pd.to_datetime(row['出院时间']) - pd.to_datetime(row['第二次手术时间'])).days
        else:
            return np.nan
    else:
        if pd.notna(row['第一次手术时间']):
            return (pd.to_datetime(row['出院时间']) - pd.to_datetime(row['第一次手术时间'])).days
        else:
            return np.nan



#######################################


#计算优先级
# 假设的优先级计算函数
def calculate_priority(row, today):
    # 初始化优先级为m0
    priority = 0

    # 获取门诊时间和类型
    clinic_date = pd.to_datetime(row['门诊时间'])
    disease_type = row['类型']

    # 计算从门诊时间到今天的天数差
    days_since_clinic = (today - clinic_date).days

    # 根据疾病类型设置优先级
    if '外伤' in disease_type:
        priority = 16
    elif '白内障(双眼)' in disease_type:
        weekday = clinic_date.weekday()
        if weekday == 5 or weekday == 6:  # 周六、周日
            priority = 15
        else:
            priority = min(days_since_clinic, 16)
    elif '白内障' in disease_type:
        weekday = clinic_date.weekday()
        if weekday in [0, 1, 5, 6]:  # 周一、周二、周六、周日
            priority = 15
        else:
            priority = min(days_since_clinic, 16)
    elif disease_type in ['青光眼', '视网膜疾病']:
        weekday = clinic_date.weekday()
        if weekday in [2, 3, 4]:  # 周三、周四、周五
            priority = 15
        else:
            priority = min(days_since_clinic, 16)
    else:
        priority = min(days_since_clinic, 16)

    return priority


# 更新数据框中未出院病人的优先级的函数
def set_priorities(df, today):
    # 遍历 DataFrame 的每一行（这里使用 iterrows，但注意它的性能可能不如向量化操作）
    for index, row in df.iterrows():
        if row['是否入院'] == '否':  # 注意这里应该是'否'，因为您要更新未出院病人的优先级
            # 使用 loc 来设置优先级
            df.at[index, '优先级'] = calculate_priority(row, today)


#设置手术时间
def set_surgery_details(df):
    # 确保日期列是datetime类型
    df['入院时间'] = pd.to_datetime(df['入院时间'], errors='coerce')

    # 定义一个辅助函数来找到最近的周几
    def find_nearest_weekday(date, weekdays):
        # weekdays应该是一个包含所需星期几（0=周一, ..., 6=周日）的列表
        days_to_add = min(
            (abs((date.weekday() - wd) % 7) + (7 if (date.weekday() - wd) % 7 < 0 else 0)) for wd in weekdays)
        return date + timedelta(days=days_to_add)

        # 遍历DataFrame中的每一行

    for index, row in df.iterrows():
        if row['是否入院'] == '是':
            admission_date = row['入院时间']
            disease_type = row['类型']

            # 设置是否手术为是
            df.at[index, '是否手术'] = '是'

            if pd.isna(admission_date):
                continue  # 如果没有入院时间，则跳过

            # 设置手术时间
            if disease_type == '外伤':
                surgery_time = admission_date + timedelta(days=1)
            elif disease_type == '白内障(双眼)':
                if admission_date.weekday() in [5, 6]:  # 周六、周日
                    surgery_time_1 = find_nearest_weekday(admission_date + timedelta(days=7), [0])  # 第二周周一
                    surgery_time_2 = find_nearest_weekday(surgery_time_1 + timedelta(days=2), [2])  # 第二周周三
                else:
                    surgery_time_1 = find_nearest_weekday(admission_date, [0, 2])  # 最近的周一或周三
                    surgery_time_2 = find_nearest_weekday(surgery_time_1 + timedelta(days=2), [0, 2])  # 最近的周一或周三
                df.at[index, '第一次手术时间'] = surgery_time_1
                df.at[index, '第二次手术时间'] = surgery_time_2
            elif disease_type == '白内障':
                if admission_date.weekday() in [0, 1]:  # 周一、周二
                    surgery_time = find_nearest_weekday(admission_date, [2])  # 最近的周三
                elif admission_date.weekday() in [5, 6]:  # 周六、周日
                    surgery_time = find_nearest_weekday(admission_date + timedelta(days=7), [0])  # 第二周周一
                else:
                    surgery_time = find_nearest_weekday(admission_date, [0, 2])  # 最近的周一或周三
            elif disease_type in ['青光眼', '视网膜疾病']:
                if admission_date.weekday() in [3, 4, 5]:  # 周三、周四、周五
                    surgery_time = admission_date + timedelta(days=2)
                else:
                    surgery_time = admission_date + timedelta(days=2)
                    if surgery_time.weekday() in [0, 2]:  # 如果落在周一或周三
                        surgery_time += timedelta(days=1)


            # 设置第一次手术时间
            if disease_type == '白内障(双眼)':
                surgery_time=surgery_time_1
            df.at[index, '第一次手术时间'] = surgery_time

            # 对于白内障(双眼)，设置第二次手术时间
            if disease_type == '白内障(双眼)':
                df.at[index, '第二次手术时间'] = surgery_time_2



#计算出院时间
def set_discharge_details(df, df_predicate, bed_availability_df):
    # 确保日期列是datetime类型
    df['入院时间'] = pd.to_datetime(df['入院时间'], errors='coerce')
    df['第一次手术时间'] = pd.to_datetime(df['第一次手术时间'], errors='coerce')
    df['第二次手术时间'] = pd.to_datetime(df['第二次手术时间'], errors='coerce')

    # 基于附件1的数据“预测”附件2的出院时间（这里简单模拟）
    # 注意：这里只是简单地从对应类型的术后观察时间分布中随机抽样一个值
    np.random.seed(0)  # 为了结果可复现

    # 遍历DataFrame中的每一行
    for index, row in df.iterrows():
        if row['是否手术'] == '是':
            disease_type = row['类型']
            surgery_date = row['第一次手术时间']
            second_surgery_date = row['第二次手术时间']

            sample_observation_day=pd.NaT
            # 计算出院时间
            if disease_type == '白内障(双眼)':
                # 对于白内障(双眼)，使用第二次手术时间
                if pd.notna(second_surgery_date):
                    #预测术后观察时间
                    observation_days = df_predicate[df_predicate['类型'] == '白内障(双眼)']['术后观察时间'].dropna()
                    sample_observation_day = np.random.choice(observation_days)
                    #出院时间
                    discharge_date = pd.to_datetime(row['第二次手术时间']) + pd.Timedelta(
                        days=sample_observation_day)
                else:
                    # 如果没有第二次手术时间，则不设置出院时间（或可以设置为NaN）
                    discharge_date = pd.NaT  # 使用pandas的NaT表示Not a Time
            else:
                # 对于其他疾病，使用第一次手术时间
                if pd.notna(surgery_date):
                    observation_days = df_predicate[df_predicate['类型'] == row['类型']]['术后观察时间'].dropna()
                    sample_observation_day = np.random.choice(observation_days)
                    discharge_date = pd.to_datetime(row['第一次手术时间']) + pd.Timedelta(
                        days=sample_observation_day)
                else:
                    # 如果没有手术时间，则不设置出院时间
                    discharge_date = pd.NaT
            # 设置出院时间和是否计算出院
            df.at[index,'预测术后观察时间']=sample_observation_day
            df.at[index, '出院时间'] = discharge_date
            df.at[index, '是否计算出院'] = '是'

#计算某天各类型病人人数
def calculate_ratio_suitable(df2_predicate,df3, target_date_str):
    # 设定目标日期
    target_date = pd.to_datetime(target_date_str)
    # 确保时间列是datetime类型
    df3['入院时间'] = pd.to_datetime(df3['入院时间'])
    df3['出院时间'] = pd.to_datetime(df3['出院时间'])

    # 计算df2_predicate中在目标日期病床占有的病人数量
    # 注意：这里我们修改条件为入院时间 <= 目标日期 且 出院时间 > 目标日期的下一天
    df2_occupied = df2_predicate[
        (df2_predicate['入院时间'] <= target_date) & (df2_predicate['出院时间'] > target_date)]
    df2_occupied_counts = df2_occupied.groupby('类型').size()

    # 计算df3中'是否计算出院'为是且'是否入院'为是且满足日期条件的病人数量
    df3_occupied_in = df3[(df3['是否计算出院'] == '是') & (df3['是否入院'] == '是') &
                          (df3['入院时间'] <= target_date) & (df3['出院时间'] > target_date)]
    df3_occupied_in_counts = df3_occupied_in.groupby('类型').size()

    # 计算df3中'是否计算出院'为是且'是否入院'为否但满足日期条件的病人数量
    df3_not_in_but_counted = df3[(df3['是否计算出院'] == '是') & (df3['是否入院'] == '否') &
                                 (df3['入院时间'] <= target_date)]
    df3_not_in_but_counted_counts = df3_not_in_but_counted.groupby('类型').size()

    # 计算并准备各个DataFrame的分组计数，重置索引以将'类型'作为普通列
    df2_occupied_counts = df2_occupied.groupby('类型').size().reset_index(name='df2_occupied')
    df3_occupied_in_counts = df3_occupied_in.groupby('类型').size().reset_index(name='df3_occupied_in')
    df3_not_in_but_counted_counts = df3_not_in_but_counted.groupby('类型').size().reset_index(
        name='df3_not_in_but_counted')

    # 合并结果，使用'类型'作为合并键
    total_occupied_counts = pd.merge(pd.merge(df2_occupied_counts, df3_occupied_in_counts, on='类型', how='outer'),
                                     df3_not_in_but_counted_counts, on='类型', how='outer').fillna(0)

    # 将NaN替换为0（如果merge没有正确处理所有情况）
    total_occupied_counts.replace(np.nan, 0, inplace=True)

    # 计算总病人数
    total_occupied_counts['病人数'] = total_occupied_counts[
        ['df2_occupied', 'df3_occupied_in', 'df3_not_in_but_counted']].sum(axis=1)

    # 如果需要，可以设置'类型'为索引
    # total_occupied_counts.set_index('类型', inplace=True)

    #print(total_occupied_counts)
    return total_occupied_counts

def process_admissions(df, today, bed_availability_df,df_proportion, df2_predicate):
    today_date = pd.to_datetime(today).date()  # 使用函数参数 today
    # 检查 today_date 是否在 bed_availability_df['日期'] 的日期中
    if today_date in bed_availability_df['日期'].dt.date.values:
        row_index = bed_availability_df[bed_availability_df['日期'].dt.date == today_date].index[0]
        available_beds = bed_availability_df.loc[row_index, '床位数']
        print(f'今天{bed_availability_df.loc[row_index, '日期']}的床位数为：{available_beds}')
    else:
        available_beds = 0
        print(f'今天{today_date}的床位数为：{available_beds}')

        # 筛选出可以入院的病人
    eligible_patients = df[(df['是否入院'] == '否') & (df['门诊时间'] <= today)].sort_values(by=['优先级', '门诊时间'],
                                                                                             ascending=[False, True])
    # 安排入院
    for _, row in eligible_patients.iterrows():
        if available_beds > 0:
            row_type=row['类型']
            available_beds_type=0
            bed_occupancy=calculate_ratio_suitable(df2_predicate,df, today)
            if bed_occupancy is not None:
                if row_type in bed_occupancy['类型']:
                    bed_count = df_proportion.loc[df_proportion['类型'] == row_type, '病床数'].values[0]
                    patient_count = bed_occupancy.loc[bed_occupancy['类型'] == row_type, '病人数'].values[0]
                    available_beds_type = bed_count - patient_count
                else:
                    patient_count = 0
                    bed_count = df_proportion.loc[df_proportion['类型'] == row_type, '病床数'].values[0]
                    available_beds_type = bed_count - patient_count
            else:
                bed_count = df_proportion.loc[df_proportion['类型'] == row_type, '病床数'].values[0]
                patient_count= bed_occupancy.loc[bed_occupancy['类型'] == row_type, '病人数'].values[0]
                available_beds_type = bed_count - patient_count

            if available_beds_type > 0:
                row['入院时间'] = today
                row['是否入院'] = '是'
                available_beds -= 1
                # 更新DataFrame
                indices_to_update = row['序号']
                df.loc[df['序号']==indices_to_update, '入院时间'] = today
                df.loc[df['序号']==indices_to_update, '是否入院'] = '是'
    # 更新床位可用性
    if today_date in bed_availability_df['日期'].dt.date.values:
        index = bed_availability_df[bed_availability_df['日期'].dt.date == today_date].index[0]
        bed_availability_df.loc[index, '床位数']=available_beds


def update_bed_availability(bed_availability_df, df_schedule):
    # 筛选出 '是否计算出院' 为 '是' 的行
    discharge_rows = df_schedule[df_schedule['是否计算出院'] == '是']
    # 提取这些行的 '出院时间'，并转换为 datetime64 类型
    discharge_dates = pd.to_datetime(discharge_rows['出院时间'])

    # 用于存储新行的DataFrame
    new_rows_df = pd.DataFrame(columns=['日期', '床位数'])

    # 更新 bed_availability_df
    for date in discharge_dates:
        # 检查 bed_availability_df 中是否存在该日期
        filtered_df = bed_availability_df[bed_availability_df['日期'] == date]
        if not filtered_df.empty:
            # 如果找到了匹配的日期，则更新床位数
            row_index = filtered_df.index[0]
            bed_availability_df.loc[row_index, '床位数'] += 1
        else:
            # 如果没有找到匹配的日期，添加一个新的日期行
            bed_availability_df = bed_availability_df._append({'日期': date, '床位数': 1}, ignore_index=True)

    return bed_availability_df#注意不添加好像不会修改bed_availability_df






####################################################################
def problem2_model(df3, df1,df_proportion, df2_predicate):
    """
    :param
    :param df3: 排队尚未入院的病人
    :param df1: 附件1用于设置术后观察时间
    :param df_proportion:病床比例
    :param df2_predicate: 表2预测术后观察时间
    :return:flag为true则说明该比例通过表2检测
    """
    # 将日期列转换为 datetime 格式，并处理无效日期
    def parse_date(date_str):
        try:
            return pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
        except Exception as e:
            print(f"日期解析错误: {date_str}，错误信息: {e}")
            return pd.NaT

    df3['门诊时间'] = df3['门诊时间'].apply(parse_date)
    df3['第二次手术时间'] = df3['第二次手术时间'].apply(parse_date)
    df3['出院时间'] = pd.to_datetime(df3['出院时间'].fillna('').replace('', np.nan), errors='coerce')
    df3['预测术后观察时间'] = pd.NaT
    # 添加新列“是否入院”，并全部设置为“否”
    df3['是否入院'] = '否'
    df3['是否手术'] = '否'
    df3['是否计算出院'] = '否'
    #########################
    df1['术后观察时间'] = df1.apply(calculate_observation_time, axis=1)
    #########################
    bed_availability_df=calculate_bed_availability(df2_predicate)
    #########################
    flag=ratio_suitable_table2(df_proportion, df2_predicate)
    if not flag:
        print('该比例不符合表2！！！！！！！！！！！！！！！！！！！！！！！！！！')
        return flag,0,0,0
    #########################
    # 初始化日期循环
    start_date = df3['门诊时间'].min()  # '附件3.xlsx'门诊时间最早为2008-08-30
    current_date = pd.to_datetime(start_date)
    # 计算'是否入院'为'否'的病人数
    not_admitted_count = df3[df3['是否入院'] == '否'].shape[0]
    print(f'尚未入院的病人数为: {not_admitted_count}')
    # 初始化尚未入院的病人数集合
    not_admitted_counts = [df3[df3['是否入院'] == '否'].shape[0]]
    # 尚未入院的病人数为0时退出循环
    while not_admitted_count:
        print('__________________________________________________________')
        print(f'今天为: {current_date}')
        set_priorities(df3, current_date)
        process_admissions(df3, current_date, bed_availability_df, df_proportion, df2_predicate)
        set_surgery_details(df3)
        # print(df3)
        set_discharge_details(df3, df1, bed_availability_df)
        bed_availability_df = update_bed_availability(bed_availability_df, df3)
        # print(bed_availability_df)
        current_date += pd.Timedelta(days=1)
        # 计算'是否入院'为'否'的病人数
        not_admitted_count = df3[df3['是否入院'] == '否'].shape[0]
        print(f'尚未入院的病人数为: {not_admitted_count}')
        not_admitted_counts.append(not_admitted_count)

    # 计算指标
    # 术前平均逗留时间
    pre_op_stay_time = (df3['第一次手术时间'] - df3['门诊时间']).mean()

    # 术前平均等待时间
    pre_op_wait_time = (df3['第一次手术时间'] - df3['入院时间']).mean()

    # 入院平均等待时间
    admission_wait_time = (df3['入院时间'] - df3['门诊时间']).mean()

    print(f"术前平均逗留时间:", pre_op_stay_time)
    print(f"术前平均等待时间:", pre_op_wait_time)
    print(f"入院平均等待时间:", admission_wait_time)
    ##########################
    print(df3)
    return True,pre_op_stay_time,pre_op_wait_time,admission_wait_time









