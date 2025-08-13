import numpy as np
import pandas as pd

def make_dataset(n_samples=800, subject_pool=120, start_idx=0, seed=42):
    np.random.seed(seed + start_idx)
    ids = np.arange(start_idx, start_idx + n_samples)
    sid = np.random.randint(1, subject_pool+1, size=n_samples)
    gender = np.random.choice(['M','F','O'], size=n_samples, p=[0.7,0.29,0.01])
    age = np.random.randint(18, 66, size=n_samples)
    height = np.random.randint(155, 196, size=n_samples)
    weight = np.round(np.random.normal(loc=75 - (age-35)*0.1, scale=8, size=n_samples),1)
    weight = np.clip(weight, 50, 120)
    hr_rest = np.round(np.random.normal(loc=60 - (age-30)*0.1, scale=5, size=n_samples)).astype(int)
    hr_rest = np.clip(hr_rest, 40, 90)
    hr_max = 208 - 0.7 * age
    hr_reserve = hr_max - hr_rest
    vt1 = np.round(hr_rest + 0.50 * hr_reserve).astype(int)
    vt2 = np.round(hr_rest + 0.85 * hr_reserve).astype(int)
    effort = np.random.beta(2,5, size=n_samples) * 0.8 + 0.15
    effort = np.clip(effort, 0.15, 0.98)
    hr_mean = np.round(hr_rest + effort * hr_reserve).astype(int)
    hr_mean = np.clip(hr_mean, hr_rest+5, np.ceil(hr_max).astype(int))
    hr_std = np.round(np.abs(np.random.normal(loc=8, scale=3, size=n_samples))).astype(int)
    hr_std = np.clip(hr_std, 2, 20)
    base_speed = np.random.normal(loc=22, scale=4, size=n_samples)
    speed_mean = np.round(base_speed + (35 - age)*0.05 + (75 - weight)*0.03 + (effort-0.4)*15, 2)
    speed_mean = np.clip(speed_mean, 8, 60)
    speed_std = np.round(np.abs(np.random.normal(loc=2.5, scale=1.5, size=n_samples)),2)
    speed_std = np.clip(speed_std, 0.2, 8)
    gain_mean = np.round(np.abs(np.random.normal(loc=120 + (effort-0.4)*400 + (age-40)*1.5, scale=140, size=n_samples)),1)
    gain_mean = np.clip(gain_mean, 0, 2500)
    gain_std = np.round(np.abs(np.random.normal(loc=40, scale=30, size=n_samples)),1)
    gain_std = np.clip(gain_std, 1, 600)
    df = pd.DataFrame({
        'idx': ids, 'sid': sid, 'gender': gender, 'age': age, 'weight': weight, 'height': height,
        'hr_rest': hr_rest, 'hr_mean': hr_mean, 'hr_std': hr_std,
        'speed_mean': speed_mean, 'speed_std': speed_std, 'gain_mean': gain_mean, 'gain_std': gain_std,
        'vt1': vt1, 'vt2': vt2
    })
    cols = ['idx','sid','gender','age','weight','height','hr_rest','hr_mean','hr_std','speed_mean','speed_std','gain_mean','gain_std','vt1','vt2']
    return df[cols]

if __name__ == '__main__':
    train = make_dataset(800, subject_pool=140, start_idx=0, seed=42)
    test = make_dataset(200, subject_pool=140, start_idx=800, seed=42)
    train.to_csv('cycling_train.csv', index=False)
    test.to_csv('cycling_test.csv', index=False)
    print('Saved cycling_train.csv and cycling_test.csv in current folder')