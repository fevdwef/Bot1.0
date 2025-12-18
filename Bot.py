import webbrowser
import time

url = input("url: ")
duration = int(input(" Time space: ")
x =int(input(" how many views: "))

for i in range(x):
  webbrowser.open_new(url)
  time.sleep(duration)
               
