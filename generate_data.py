# Generates the same fake CSVs as provided
import numpy as np
import pandas as pd
import os

os.makedirs("data", exist_ok=True)
T=240; fs=50
t=np.arange(0,T,1/fs)
N=len(t); N1=N//3; N2=2*N//3
step_freq=np.zeros_like(t)
step_freq[:N1]=1.6; step_freq[N1:N2]=2.2; step_freq[N2:]=3.0
phase=2*np.pi*np.cumsum(step_freq)/fs
acc_z=0.5+1.0*np.sin(phase)+0.2*np.random.randn(len(t))
acc_x=0.05*np.random.randn(len(t))
acc_y=0.05*np.random.randn(len(t))
df_acc=pd.DataFrame({
    "Time [s]":t,
    "Linear Acceleration x [m/s^2]":acc_x,
    "Linear Acceleration y [m/s^2]":acc_y,
    "Linear Acceleration z [m/s^2]":acc_z
})
df_acc.to_csv("data/Accelerometer.csv",index=False)

lat0,lon0=60.0,24.0
speed=np.zeros_like(t)
speed[:N1]=1.6; speed[N1:N2]=2.2; speed[N2:]=3.0
dt=1/fs
dist=np.cumsum(speed*dt)
lat=lat0+(dist/111000.0)
lon=np.full_like(lat, lon0)
df_loc=pd.DataFrame({
    "Time [s]":t,
    "Latitude [deg]":lat,
    "Longitude [deg]":lon,
    "Speed [m/s]":speed
})
df_loc.to_csv("data/Location.csv",index=False)
print("Done.")
