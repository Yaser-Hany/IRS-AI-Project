import numpy as np
import pandas as pd

n_samples = 1000

# 1. تحديد إحداثيات ثابتة لمحطة الإرسال (Tx) والـ IRS (بالمتر)
tx_pos = np.array([0, 0, 5])    # المحطة في نقطة الأصل وارتفاع 5 متر
irs_pos = np.array([10, 10, 3]) # الـ IRS على بعد 10 متر وارتفاع 3 متر

# 2. توليد إحداثيات عشوائية لـ 1000 مستخدم (Rx)
rx_x = np.random.uniform(0, 30, n_samples)  # المحور X للمستخدمين
rx_y = np.random.uniform(0, 30, n_samples)  # المحور Y للمستخدمين
rx_z = np.random.uniform(1.0, 1.8, n_samples) # ارتفاع الهاتف في يد المستخدم (Z)

# 3. حساب المسافة ثلاثية الأبعاد (Euclidean Distance)
# المسافة الثابتة من المحطة إلى الـ IRS
dist_tx_irs = np.sqrt((irs_pos[0]-tx_pos[0])**2 + (irs_pos[1]-tx_pos[1])**2 + (irs_pos[2]-tx_pos[2])**2)

# المسافة المتغيرة من الـ IRS لكل مستخدم
dist_irs_rx = np.sqrt((rx_x - irs_pos[0])**2 + (rx_y - irs_pos[1])**2 + (rx_z - irs_pos[2])**2)

# المسافة الكلية التي تقطعها الإشارة
total_distance = dist_tx_irs + dist_irs_rx

# 4. بناء الداتا فريم الجديدة بالـ 3D Features
patterns = np.random.randint(0, 5, n_samples)
X_irs_3d = pd.DataFrame({
    'Rx_X': rx_x,
    'Rx_Y': rx_y,
    'Rx_Z': rx_z,
    'Total_Distance': total_distance,
    'Pattern': patterns
})

# 5. محاكاة قوة الإشارة بناءً على المسافة الحقيقية ثلاثية الأبعاد
# الإشارة تضعف كلما زادت المسافة الكلية (Path Loss)
y_irs_3d = -30 - 20 * np.log10(total_distance) + 2 * patterns + np.random.randn(n_samples) * 2

print(X_irs_3d.head())