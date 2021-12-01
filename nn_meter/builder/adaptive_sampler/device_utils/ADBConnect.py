import subprocess,re

class ADBConnect:
    def __init__(self, serial=None):   
        devices = subprocess.check_output(f'adb devices', shell=True).decode('utf-8')
        device_list = re.findall(r'([a-zA-Z0-9]+)[^\w]*([a-zA-Z0-9]+)', devices.split('List of devices attached')[-1])
        if serial == None:
            if len(device_list) == 0:
                raise FileNotFoundError
            else:
                self.serial = device_list[0][0]
                print(f'Device {self.serial} selected.')
        else:
           # print(device_list,serial)
            for device in device_list:
                if serial == device[0]:
                    self.serial = serial
                    print(f'Device {self.serial} here selected.')
                    return
            raise FileNotFoundError

    def push_files(self, src, dst):
        subprocess.check_output(f'adb -s {self.serial} push {src} {dst}', shell=True)
    
    def pull_files(self, src, dst):
        subprocess.check_output(f'adb -s {self.serial} pull {src} {dst}', shell=True)
    
    def run_cmd(self, cmd):
        #print(self.serial)
        results = subprocess.check_output(f'adb -s {self.serial} shell su -c {cmd}', shell=True).decode('utf-8')
        #print(results)
        #latency=get_avg_latency(results)
        #print(latency)

        return results