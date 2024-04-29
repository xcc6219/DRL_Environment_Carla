import subprocess

CARLA_PATH = "D:\\Project\\carla\\WindowsNoEditor"
# CARLA_PATH = 'D:\Project\carla\WindowsNoEditor-0.9.10'
# subprocess.Popen('CarlaUE4.exe -quality-level=Low -carla-port=2000 -no-rendering' , cwd=CARLA_PATH, shell=True)
# subprocess.call("E:\\carla0911\\WindowsNoEditor\\CarlaUE4.exe -quality-level=Low")
subprocess.Popen('CarlaUE4.exe -carla-port=2000 -ResX=980 -ResY=540',
                 cwd=CARLA_PATH, shell=True)
