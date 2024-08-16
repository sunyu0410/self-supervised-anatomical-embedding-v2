import time

class FileMonitor():
    def __init__(self, filepath, respond:callable, time_interval=1):
        self.filepath = filepath
        self.last = self.read()
        self.time_interval = time_interval
        self.respond = respond
    
    def read(self):
        return open(self.filepath).read().strip()
    
    def start(self):
        print(f'Monitoring: {self.filepath}')
        while True:
            time.sleep(self.time_interval)

            current = self.read()
            if current == self.last:
                continue
            else:
                self.respond()
                self.last = current