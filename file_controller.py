import os
def get_start_day(name):
    with open(name) as f:
        for i, line in enumerate(f):
            if i == 4:
                date_str = line.split()[2]  # 获取第五行中日期字符串
                time_str = line.split()[3]
                start_date = date_str.split()[-1].strip()[:] + " " + time_str[:8]
                start_day = int(start_date[8:10])
                start_hour = int(time_str[:2]) + round(int(time_str[3:5]) / 60, 1)
                break
        return start_day, start_hour


def cal_begin_days(start_day, start_hour):
    end_day = 28
    end_hour = 24
    gap_day = end_day - start_day
    gap_hour = end_hour - start_hour
    if gap_day < 0:
        gap_day = 0
        gap_hour = 0
    return round(gap_day + gap_hour / 24, 1)

def get_begin_days(name):
    start_day, start_hour = get_start_day(name)
    begin_days = cal_begin_days(start_day, start_hour)
    return begin_days

def copy_file(src, dst):
    with open(src, 'r') as f:
        lines = f.readlines()
        with open(dst, 'w') as f:
            f.seek(0)
            f.truncate()
            for line in lines:
                f.write(line)

def get_max_rho_file(path):
    #获取所有后缀为.rho的文件
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.rho')]
    files.sort(key=lambda fn: os.path.getmtime(path + "\\" + fn))
    file = os.path.join(path, files[-1])
    return file

def get_edi_files(path):
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.edi')]
    return files
