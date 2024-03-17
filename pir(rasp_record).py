import RPi.GPIO as GPIO
import time
import datetime
import http.server
import socketserver
import threading

'''
Program for record PIR (motion sensor) data
starts on raspberry_pi connected to PIR

Specify the GPIO number of pir connection
'''
PIR_PIN = 18

#class MyHandler(http.server.SimpleHTTPRequestHandler):
    #def do_GET(self):
        #global trig_dataset
        #if self.path == "/datetime":
            #self.send_response(200)
            #self.send_header('Content-type', 'teext/html')
            #self.end_headers()
            #data = "\n".join(trig_dataset)
            #self.wfile.write(data.encode('utf-8'))
            #return
        #else:
            #super().do_GET()

def pir_motion_handler(channel):
    global trig_dataset
    print('MOTION!')
    trig_datetime = datetime.datetime.now()
    trig_datetime = trig_datetime.strftime('%Y-%m-%d %H:%M:%S')
    print(trig_datetime, "\n")
    trig_dataset.append(trig_datetime)

    with open(name_file, "a") as log_file:
        log_file.write(trig_datetime + "\n")    

#PORT = 8080

if __name__=='__main__':
    name_file =  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "pir_log.txt"
    with open(name_file, "a") as log_file:
        pass
    pir_pin=PIR_PIN
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pir_pin, GPIO.IN)

    trig_dataset = []
    GPIO.add_event_detect(pir_pin, GPIO.RISING, callback=pir_motion_handler)
    #httpd = socketserver.TCPServer(("", PORT), MyHandler)
    #server_thread = threading.Thread(target=httpd.serve_forever)
    #server_thread.daemon = True
    #server_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('INTERRUPT')
        GPIO.cleanup()
        #httpd.shutdown()